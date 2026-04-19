"""RL environment for the Unitree B2 quadruped that chases a virtual target.

Mirrors the structure of ``Genesis/examples/locomotion/go2_env.py`` but:

* Loads the **Unitree B2** URDF from ``unitree_ros/robots/b2_description``
  (the same asset already used by ``src/example_b2_teleop.py``).
* Drops the velocity-tracking rewards. In their place the robot is rewarded for

    - being close to a "red ball" placed at a small random distance around it
      (``_reward_tracking_target``), and
    - aligning its body yaw with the commanded target yaw
      (``_reward_tracking_yaw``).

* Exposes the target in the observation as ``(x_body, y_body, cos(yaw_err),
  sin(yaw_err))`` — i.e. in the robot body frame so the policy is invariant
  to the world frame.
* Keeps the single dynamic coefficient ``target_coeff`` that ``b2_train.py``
  ramps from ``0`` up to ``max_target_coeff`` once locomotion has stabilised
  (plateau detector, same as ``dynamic_ics_dog_train.py``).
* Adds ``enable_external_target`` + ``set_external_target`` so inference code
  (``b2_eval.py``, ``api.py``, ``ds4_control.py``) can drive the target from
  outside without the training-time random resampling.
"""

from __future__ import annotations

import math
import os

import torch

import genesis as gs
from genesis.utils.geom import (
    inv_quat,
    quat_to_xyz,
    transform_by_quat,
    transform_quat_by_quat,
)


B2_URDF_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "unitree_ros",
        "robots",
        "b2_description",
        "urdf",
        "b2_description.urdf",
    )
)


def gs_rand(lower: torch.Tensor, upper: torch.Tensor, batch_shape) -> torch.Tensor:
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(
        size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device
    ) + lower


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap an angle tensor to ``[-pi, pi]``."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class B2TargetEnv:
    """Target-tracking RL environment for Unitree B2.

    Observation layout (``num_obs = 46``):

    =============================================  ===
    ``base_ang_vel * ang_vel_scale``                 3
    ``projected_gravity``                            3
    ``target_rel_body_xy * target_pos_scale``        2
    ``cos(yaw_err), sin(yaw_err)``                   2
    ``(dof_pos - default_dof_pos) * dof_pos_scale`` 12
    ``dof_vel * dof_vel_scale``                     12
    ``last_actions``                                12
    =============================================  ===
    """

    def __init__(
        self,
        num_envs: int,
        env_cfg: dict,
        obs_cfg: dict,
        reward_cfg: dict,
        command_cfg: dict,
        target_cfg: dict | None = None,
        show_viewer: bool = False,
    ):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg.get("num_commands", 0)
        self.device = gs.device

        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = env_cfg.get("dt", 0.02)
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        target_cfg = target_cfg or {}
        self.target_min_dist = target_cfg.get("min_dist", 0.3)
        self.target_max_dist = target_cfg.get("max_dist", 1.2)
        self.target_reach_dist = target_cfg.get("reach_dist", 0.20)
        self.target_reach_yaw = target_cfg.get("reach_yaw", 0.25)
        self.target_resample_steps = int(
            target_cfg.get("resample_time_s", 8.0) / self.dt
        )
        self.target_height = target_cfg.get("height", env_cfg["base_init_pos"][2])
        self.target_random_yaw = target_cfg.get("random_yaw", True)

        # Dynamic scheduling coefficient. Phase 1: 0 (robot only learns to
        # stand). Phase 2: ramped up so target rewards start dominating.
        self.target_coeff = 0.0

        # Toggle for inference: when True the env stops resampling targets
        # and instead follows whatever `set_external_target` writes.
        self.external_target_enabled = False

        # ----------------------------- Scene --------------------------------
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=20,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.8, 1.8, 1.5),
                camera_lookat=(0.0, 0.0, 0.4),
                camera_fov=45,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
        )

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=B2_URDF_PATH,
                pos=tuple(env_cfg["base_init_pos"]),
                quat=tuple(env_cfg["base_init_quat"]),
                merge_fixed_links=True,
            ),
        )

        self.scene.build(n_envs=num_envs)

        # --------------------------- Joints ---------------------------------
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=self.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # ---------------------- Fixed constants -----------------------------
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], dtype=gs.tc_float, device=self.device
        )
        self.init_base_pos = torch.tensor(
            env_cfg["base_init_pos"], dtype=gs.tc_float, device=self.device
        )
        self.init_base_quat = torch.tensor(
            env_cfg["base_init_quat"], dtype=gs.tc_float, device=self.device
        )
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][j.name] for j in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=self.device,
        )
        self.init_qpos = torch.concatenate(
            (self.init_base_pos, self.init_base_quat, self.init_dof_pos)
        )
        self.init_projected_gravity = transform_by_quat(
            self.global_gravity, self.inv_base_init_quat
        )

        # --------------------------- Buffers --------------------------------
        self.base_lin_vel = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.base_ang_vel = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.projected_gravity = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.obs_buf = torch.zeros((num_envs, self.num_obs), dtype=gs.tc_float, device=self.device)
        self.rew_buf = torch.zeros((num_envs,), dtype=gs.tc_float, device=self.device)
        self.reset_buf = torch.ones((num_envs,), dtype=gs.tc_bool, device=self.device)
        self.episode_length_buf = torch.zeros((num_envs,), dtype=gs.tc_int, device=self.device)
        self.actions = torch.zeros((num_envs, self.num_actions), dtype=gs.tc_float, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = self.init_base_pos.unsqueeze(0).expand(num_envs, -1).contiguous()
        self.base_quat = self.init_base_quat.unsqueeze(0).expand(num_envs, -1).contiguous()
        self.base_euler = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.base_yaw = torch.zeros((num_envs,), dtype=gs.tc_float, device=self.device)
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["joint_names"]],
            dtype=gs.tc_float,
            device=self.device,
        )

        # ---------------------------- Target --------------------------------
        self.target_pos_world = torch.zeros(
            (num_envs, 3), dtype=gs.tc_float, device=self.device
        )
        self.target_pos_world[:, 2] = self.target_height
        self.target_yaw = torch.zeros(
            (num_envs,), dtype=gs.tc_float, device=self.device
        )
        self.target_step_counter = torch.zeros(
            (num_envs,), dtype=gs.tc_int, device=self.device
        )
        self.target_rel_body = torch.zeros_like(self.base_pos)
        self.yaw_error = torch.zeros_like(self.base_yaw)

        self.extras: dict = {"observations": {}}

        # --------------------------- Rewards --------------------------------
        self.reward_functions, self.episode_sums = {}, {}
        for name in list(self.reward_scales.keys()):
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (num_envs,), dtype=gs.tc_float, device=self.device
            )

        self._spawn_targets_from_init()
        self._recompute_target_rel()

    # =====================================================================
    # External target control (used by API / DS4 / arrow-key eval)
    # =====================================================================
    def enable_external_target(self, enabled: bool = True) -> None:
        self.external_target_enabled = enabled

    def set_external_target(
        self,
        pos_xy: tuple[float, float] | torch.Tensor,
        yaw: float | torch.Tensor = 0.0,
        env_idx: int | None = None,
    ) -> None:
        """Override the world-frame target position / yaw."""
        pos_xy_t = torch.as_tensor(pos_xy, dtype=gs.tc_float, device=self.device)
        yaw_t = torch.as_tensor(yaw, dtype=gs.tc_float, device=self.device)
        if env_idx is None:
            self.target_pos_world[:, 0] = pos_xy_t[0]
            self.target_pos_world[:, 1] = pos_xy_t[1]
            self.target_pos_world[:, 2] = self.target_height
            self.target_yaw.fill_(float(yaw_t))
        else:
            self.target_pos_world[env_idx, 0] = pos_xy_t[0]
            self.target_pos_world[env_idx, 1] = pos_xy_t[1]
            self.target_pos_world[env_idx, 2] = self.target_height
            self.target_yaw[env_idx] = float(yaw_t)
        self._recompute_target_rel()

    # =====================================================================
    # Target spawning during training
    # =====================================================================
    def _random_target_offsets(self, n: int):
        dist = (
            self.target_min_dist
            + (self.target_max_dist - self.target_min_dist)
            * torch.rand(n, device=self.device, dtype=gs.tc_float)
        )
        theta = (
            2.0 * math.pi
            * torch.rand(n, device=self.device, dtype=gs.tc_float)
            - math.pi
        )
        dx = dist * torch.cos(theta)
        dy = dist * torch.sin(theta)
        if self.target_random_yaw:
            tyaw = (
                2.0 * math.pi
                * torch.rand(n, device=self.device, dtype=gs.tc_float)
                - math.pi
            )
        else:
            tyaw = torch.zeros(n, device=self.device, dtype=gs.tc_float)
        return dx, dy, tyaw

    def _spawn_targets_from_init(self) -> None:
        dx, dy, tyaw = self._random_target_offsets(self.num_envs)
        self.target_pos_world[:, 0] = self.init_base_pos[0] + dx
        self.target_pos_world[:, 1] = self.init_base_pos[1] + dy
        self.target_pos_world[:, 2] = self.target_height
        self.target_yaw = tyaw

    def _spawn_targets(self, envs_idx: torch.Tensor | None) -> None:
        """Random-target resample used from ``_reset_idx``.

        ``envs_idx`` may be either ``None`` (resample every env) or a **bool**
        mask over all envs (used from the step() loop to resample the envs
        that just reached / timed out).
        """
        if self.external_target_enabled:
            return
        if envs_idx is None:
            self._spawn_targets_from_init()
            return
        if envs_idx.dtype == torch.bool:
            if not envs_idx.any():
                return
            dx, dy, tyaw = self._random_target_offsets(self.num_envs)
            new_x = self.base_pos[:, 0] + dx
            new_y = self.base_pos[:, 1] + dy
            self.target_pos_world[:, 0] = torch.where(envs_idx, new_x, self.target_pos_world[:, 0])
            self.target_pos_world[:, 1] = torch.where(envs_idx, new_y, self.target_pos_world[:, 1])
            self.target_pos_world[:, 2] = self.target_height
            self.target_yaw = torch.where(envs_idx, tyaw, self.target_yaw)
        else:
            self._spawn_targets_at_indices(envs_idx)

    def _spawn_targets_at_indices(self, envs_idx: torch.Tensor) -> None:
        if self.external_target_enabled:
            return
        n = int(envs_idx.numel())
        if n == 0:
            return
        dx, dy, tyaw = self._random_target_offsets(n)
        self.target_pos_world[envs_idx, 0] = self.base_pos[envs_idx, 0] + dx
        self.target_pos_world[envs_idx, 1] = self.base_pos[envs_idx, 1] + dy
        self.target_pos_world[envs_idx, 2] = self.target_height
        self.target_yaw[envs_idx] = tyaw

    # =====================================================================
    # Observation helpers
    # =====================================================================
    def _recompute_target_rel(self) -> None:
        rel_world = self.target_pos_world - self.base_pos
        inv_q = inv_quat(self.base_quat)
        self.target_rel_body = transform_by_quat(rel_world, inv_q)
        self.yaw_error = wrap_angle(self.target_yaw - self.base_yaw)

    # =====================================================================
    # Core RL interface
    # =====================================================================
    def step(self, actions: torch.Tensor):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        self.base_yaw = torch.deg2rad(self.base_euler[:, 2])
        inv_base_q = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_q)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_q)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_q)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        self._recompute_target_rel()

        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.target_step_counter += 1
        dist_xy = torch.linalg.norm(self.target_rel_body[:, :2], dim=1)
        reached = (dist_xy < self.target_reach_dist) & (
            torch.abs(self.yaw_error) < self.target_reach_yaw
        )
        timeout = self.target_step_counter >= self.target_resample_steps
        respawn_mask = reached | timeout
        if respawn_mask.any():
            self._spawn_targets(respawn_mask)
            self.target_step_counter = torch.where(
                respawn_mask,
                torch.zeros_like(self.target_step_counter),
                self.target_step_counter,
            )

        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        # Catch NaN / exploded envs and recycle them - mirrors go2_env.py.
        try:
            self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()
        except AttributeError:
            pass

        self.extras["time_outs"] = (
            self.episode_length_buf > self.max_episode_length
        ).to(dtype=gs.tc_float)

        reset_idx = self.reset_buf.nonzero(as_tuple=False).flatten()
        if reset_idx.numel() > 0:
            self._reset_idx(reset_idx)
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset(self):
        self._reset_idx(None)
        self._update_observation()
        return self.obs_buf, None

    def _reset_idx(self, envs_idx: torch.Tensor | None):
        """Reset the selected environments.

        ``envs_idx`` is either ``None`` (reset every env) or an integer tensor
        of environment ids to reset (``nonzero().flatten()``-style).
        """
        if envs_idx is None:
            n = self.num_envs
            idx_all = torch.arange(n, device=self.device, dtype=gs.tc_int)
            self._apply_reset(idx_all, n)
            # Reset bookkeeping for all envs.
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.default_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
            self.base_yaw.zero_()
            self.target_step_counter.zero_()

            self.extras["episode"] = {}
            for key, value in self.episode_sums.items():
                self.extras["episode"]["rew_" + key] = (
                    value.mean() / self.env_cfg["episode_length_s"]
                )
                value.zero_()

            self._spawn_targets(None)
            self._recompute_target_rel()
            return

        n = int(envs_idx.numel())
        if n == 0:
            return

        self._apply_reset(envs_idx, n)
        self.base_pos[envs_idx] = self.init_base_pos
        self.base_quat[envs_idx] = self.init_base_quat
        self.projected_gravity[envs_idx] = self.init_projected_gravity
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.base_lin_vel[envs_idx] = 0.0
        self.base_ang_vel[envs_idx] = 0.0
        self.dof_vel[envs_idx] = 0.0
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.base_yaw[envs_idx] = 0.0

        self.extras["episode"] = {}
        denom = float(n) * self.env_cfg["episode_length_s"]
        for key, value in self.episode_sums.items():
            self.extras["episode"]["rew_" + key] = value[envs_idx].sum() / denom
            value[envs_idx] = 0.0

        self._spawn_targets_at_indices(envs_idx)
        self.target_step_counter[envs_idx] = 0
        self._recompute_target_rel()

    def _apply_reset(self, envs_idx: torch.Tensor, n: int) -> None:
        """Write base pose + joint defaults into the simulator."""
        self.robot.set_pos(
            self.init_base_pos.unsqueeze(0).expand(n, -1).contiguous(),
            envs_idx=envs_idx,
        )
        self.robot.set_quat(
            self.init_base_quat.unsqueeze(0).expand(n, -1).contiguous(),
            envs_idx=envs_idx,
        )
        self.robot.set_dofs_position(
            position=self.default_dof_pos.unsqueeze(0).expand(n, -1).contiguous(),
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        try:
            self.robot.zero_all_dofs_velocity(envs_idx=envs_idx)
        except TypeError:
            self.robot.zero_all_dofs_velocity()

    def _update_observation(self):
        cos_yaw = torch.cos(self.yaw_error).unsqueeze(1)
        sin_yaw = torch.sin(self.yaw_error).unsqueeze(1)
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],                 # 3
                self.projected_gravity,                                          # 3
                self.target_rel_body[:, :2] * self.obs_scales["target_pos"],     # 2
                cos_yaw,                                                         # 1
                sin_yaw,                                                         # 1
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],                       # 12
                self.actions,                                                    # 12
            ),
            dim=-1,
        )

    # =====================================================================
    # Reward functions
    # =====================================================================
    def _reward_tracking_target(self):
        """Bell-shaped reward on XY distance to the target, scaled by coeff."""
        dist_sq = torch.sum(self.target_rel_body[:, :2] ** 2, dim=1)
        sigma = self.reward_cfg.get("target_sigma", 0.5)
        return self.target_coeff * torch.exp(-dist_sq / sigma)

    def _reward_tracking_yaw(self):
        """Bell-shaped reward on yaw error, scaled by coeff."""
        sigma = self.reward_cfg.get("yaw_sigma", 0.5)
        return self.target_coeff * torch.exp(-self.yaw_error ** 2 / sigma)

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])


# =========================================================================
# Default configuration for B2
# =========================================================================
def default_cfgs():
    env_cfg = {
        "num_actions": 12,
        # Stand pose consistent with src/example_b2_teleop.py.
        "default_joint_angles": {
            "FR_hip_joint": 0.0, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
            "FL_hip_joint": 0.0, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
            "RR_hip_joint": 0.0, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
            "RL_hip_joint": 0.0, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        # Stiffer than the Go2 defaults (20/0.5) because B2 is a much
        # heavier robot. Derived loosely from
        # unitree_ros/robots/b2_description/config/robot_control.yaml.
        "kp": 200.0,
        "kd": 5.0,
        "termination_if_roll_greater_than": 25,   # deg
        "termination_if_pitch_greater_than": 25,
        "base_init_pos": [0.0, 0.0, 0.62],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "dt": 0.02,
    }
    obs_cfg = {
        "num_obs": 46,
        "obs_scales": {
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "target_pos": 1.0,
        },
    }
    reward_cfg = {
        "base_height_target": 0.55,
        "target_sigma": 0.5,
        "yaw_sigma": 0.5,
        "reward_scales": {
            # ---- Stability (Phase 1) ----
            "lin_vel_z": -1.5,
            "ang_vel_xy": -0.05,
            "orientation": -3.0,
            "base_height": -30.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            # ---- Target tracking (ramped up via target_coeff in Phase 2) ----
            "tracking_target": 2.0,
            "tracking_yaw": 1.0,
        },
    }
    command_cfg = {"num_commands": 0}
    target_cfg = {
        "min_dist": 0.3,
        "max_dist": 1.2,
        "reach_dist": 0.20,
        "reach_yaw": 0.25,
        "resample_time_s": 8.0,
        "height": 0.55,
        "random_yaw": True,
    }
    dynamic_cfg = {
        "plateau_window": 50,
        "plateau_threshold": 0.07,
        "plateau_patience": 20,
        "coeff_increment": 0.07,
        "max_target_coeff": 1.0,
        "check_interval": 5,
        "min_iterations": 50,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg, target_cfg, dynamic_cfg
