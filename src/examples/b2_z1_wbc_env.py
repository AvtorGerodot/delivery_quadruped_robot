"""Whole-body control RL environment for the Unitree B2 + Z1 robot.

This is the *mobile manipulation* counterpart of ``b2_env.py``. The earlier
``B2TargetEnv`` rewarded the **base** for chasing a virtual ball expressed in
the robot's *body* frame. Here, instead:

* The robot is the combined **B2 + Z1** (``complex_urdf/b2_z1.urdf``); the arm
  is **unfrozen** and added to the action space, so the policy controls all
  18 DOFs (12 legs + 6 arm) for true whole-body coordination.
* The red ball lives in the **world** frame and is 3-D ``(x, y, z)``: it spawns
  at a small random XY distance around the robot (same neighbourhood as
  ``b2_env.py``) and at a random height **inside the arm's reachable band**.
* The tracking reward penalises the distance between the **manipulator's
  end-effector** (the ``z1_link06`` flange + a small tool offset) and the ball
  — *not* the base centre of mass. The base must therefore walk close enough
  for the arm to reach, and the arm must extend to touch the ball.
* A **self-hit penalty** keeps the end-effector out of a keep-out sphere around
  the trunk, discouraging the arm from folding into / striking the body.

Reward groups
-------------
Stability (always on): ``lin_vel_z``, ``ang_vel_xy``, ``orientation``,
``base_height``, ``action_rate``, ``similar_to_default`` (legs only),
``arm_posture`` (arm to home), ``self_collision`` (keep-out sphere).

Tracking (ramped up via ``target_coeff`` in Phase 2 by ``b2_z1_wbc_train.py``):
``tracking_ee`` — bell-shaped on the 3-D EE→ball distance.

Observation layout (``num_obs = 66``)::

    base_ang_vel * ang_vel_scale                3
    projected_gravity                           3
    ball_rel_base   (body frame) * scale        3
    ee_to_ball      (body frame) * scale        3
    (dof_pos - default_dof_pos) * dof_pos_scale 18
    dof_vel * dof_vel_scale                     18
    last_actions                                18
    -------------------------------------------- --
    total                                       66

The reset / aliasing workarounds are inherited from ``b2_env.py`` (explicit
``set_pos`` + ``set_quat`` + ``set_dofs_position``, ``.clone()`` on expanded
init tensors).
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


B2_Z1_URDF_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "complex_urdf",
        "b2_z1.urdf",
    )
)


def gs_rand(lower: torch.Tensor, upper: torch.Tensor, batch_shape) -> torch.Tensor:
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(
        size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device
    ) + lower


class B2Z1WholeBodyEnv:
    """Whole-body (legs + arm) end-effector reaching RL environment."""

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

        # ---- Joint groups: legs (controlled for locomotion) + arm (reaching).
        self.leg_joint_names = list(env_cfg["leg_joint_names"])
        self.arm_joint_names = list(env_cfg["arm_joint_names"])
        self.joint_names = self.leg_joint_names + self.arm_joint_names
        self.n_leg = len(self.leg_joint_names)
        self.n_arm = len(self.arm_joint_names)
        assert self.num_actions == self.n_leg + self.n_arm

        # ---- Target config -------------------------------------------------
        target_cfg = target_cfg or {}
        self.target_min_dist = target_cfg.get("min_dist", 0.3)
        self.target_max_dist = target_cfg.get("max_dist", 1.2)
        self.target_z_min = target_cfg.get("z_min", 0.25)
        self.target_z_max = target_cfg.get("z_max", 0.85)
        self.target_reach_dist = target_cfg.get("reach_dist", 0.15)
        self.target_resample_steps = int(
            target_cfg.get("resample_time_s", 8.0) / self.dt
        )
        self.ee_link_name = target_cfg.get("ee_link", "z1_link06")
        ee_offset = target_cfg.get("ee_offset", (0.08, 0.0, 0.0))
        self.ee_offset = torch.tensor(
            ee_offset, dtype=gs.tc_float, device=self.device
        )
        self.keepout_radius = target_cfg.get("keepout_radius", 0.20)

        # Dynamic scheduling coefficient (ramped 0 -> max in Phase 2).
        self.target_coeff = 0.0
        # Inference toggle: stop random resampling, follow set_external_target.
        self.external_target_enabled = False

        # ----------------------------- Scene --------------------------------
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=20,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.8, 1.8, 1.6),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=45,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
        )
        urdf_path = env_cfg.get("urdf_path", B2_Z1_URDF_PATH)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                pos=tuple(env_cfg["base_init_pos"]),
                quat=tuple(env_cfg["base_init_quat"]),
                merge_fixed_links=True,
            ),
        )
        self.scene.build(n_envs=num_envs)

        # --------------------------- Joint indexing --------------------------
        # Index sim DOFs directly (legs and arm DOFs are interleaved in the
        # combined URDF, so a contiguous slice would be wrong).
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.joint_names],
            dtype=gs.tc_int,
            device=self.device,
        )

        # Per-DOF PD gains: stiff legs, lighter arm.
        kp = [env_cfg["leg_kp"]] * self.n_leg + [env_cfg["arm_kp"]] * self.n_arm
        kd = [env_cfg["leg_kd"]] * self.n_leg + [env_cfg["arm_kd"]] * self.n_arm
        self.robot.set_dofs_kp(kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(kd, self.motors_dof_idx)

        # Per-DOF action scale: arm gets a wider scale so it can sweep its
        # workspace within a reasonable action magnitude.
        leg_scale = env_cfg["action_scale"]
        arm_scale = env_cfg.get("arm_action_scale", leg_scale)
        self.action_scale = torch.tensor(
            [leg_scale] * self.n_leg + [arm_scale] * self.n_arm,
            dtype=gs.tc_float,
            device=self.device,
        )

        # End-effector link handle (survives merge_fixed_links: revolute joint).
        self.ee_link = self.robot.get_link(self.ee_link_name)

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
        self.init_projected_gravity = transform_by_quat(
            self.global_gravity, self.inv_base_init_quat
        )
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in self.joint_names],
            dtype=gs.tc_float,
            device=self.device,
        )
        self.default_arm_pos = self.default_dof_pos[self.n_leg :]

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
        # `.clone()` (not `.contiguous()`) to avoid the num_envs==1 aliasing bug.
        self.base_pos = self.init_base_pos.unsqueeze(0).expand(num_envs, -1).clone()
        self.base_quat = self.init_base_quat.unsqueeze(0).expand(num_envs, -1).clone()
        self.base_euler = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.ee_pos = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=self.device)

        # ---------------------------- Target --------------------------------
        self.target_pos_world = torch.zeros(
            (num_envs, 3), dtype=gs.tc_float, device=self.device
        )
        self.target_step_counter = torch.zeros(
            (num_envs,), dtype=gs.tc_int, device=self.device
        )
        # Body-frame relations (filled by _recompute_target_rel()).
        self.ball_rel_body = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.ee_to_ball_body = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=self.device)
        self.ee_to_ball_world = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=self.device)

        self.extras: dict = {"observations": {}}

        # --------------------------- Rewards --------------------------------
        self.reward_functions, self.episode_sums = {}, {}
        for name in list(self.reward_scales.keys()):
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (num_envs,), dtype=gs.tc_float, device=self.device
            )

        self._update_ee_pos()
        self._spawn_targets_from_init()
        self._recompute_target_rel()

    # =====================================================================
    # External target control (used by eval / API)
    # =====================================================================
    def enable_external_target(self, enabled: bool = True) -> None:
        self.external_target_enabled = enabled

    def set_external_target(
        self,
        pos_xyz: tuple[float, float, float] | torch.Tensor,
        env_idx: int | None = None,
    ) -> None:
        """Override the world-frame 3-D target ball position."""
        pos = torch.as_tensor(pos_xyz, dtype=gs.tc_float, device=self.device)
        if env_idx is None:
            self.target_pos_world[:, 0] = pos[0]
            self.target_pos_world[:, 1] = pos[1]
            self.target_pos_world[:, 2] = pos[2]
        else:
            self.target_pos_world[env_idx] = pos
        self._recompute_target_rel()

    # =====================================================================
    # Target spawning during training
    # =====================================================================
    def _random_target(self, n: int):
        dist = (
            self.target_min_dist
            + (self.target_max_dist - self.target_min_dist)
            * torch.rand(n, device=self.device, dtype=gs.tc_float)
        )
        theta = (
            2.0 * math.pi * torch.rand(n, device=self.device, dtype=gs.tc_float)
            - math.pi
        )
        dx = dist * torch.cos(theta)
        dy = dist * torch.sin(theta)
        z = (
            self.target_z_min
            + (self.target_z_max - self.target_z_min)
            * torch.rand(n, device=self.device, dtype=gs.tc_float)
        )
        return dx, dy, z

    def _spawn_targets_from_init(self) -> None:
        dx, dy, z = self._random_target(self.num_envs)
        self.target_pos_world[:, 0] = self.init_base_pos[0] + dx
        self.target_pos_world[:, 1] = self.init_base_pos[1] + dy
        self.target_pos_world[:, 2] = z

    def _spawn_targets(self, mask: torch.Tensor | None) -> None:
        """Resample targets. ``mask`` is None (all) or a bool mask over envs."""
        if self.external_target_enabled:
            return
        if mask is None:
            self._spawn_targets_from_init()
            return
        if not mask.any():
            return
        dx, dy, z = self._random_target(self.num_envs)
        new_x = self.base_pos[:, 0] + dx
        new_y = self.base_pos[:, 1] + dy
        self.target_pos_world[:, 0] = torch.where(mask, new_x, self.target_pos_world[:, 0])
        self.target_pos_world[:, 1] = torch.where(mask, new_y, self.target_pos_world[:, 1])
        self.target_pos_world[:, 2] = torch.where(mask, z, self.target_pos_world[:, 2])

    def _spawn_targets_at_indices(self, envs_idx: torch.Tensor) -> None:
        if self.external_target_enabled:
            return
        n = int(envs_idx.numel())
        if n == 0:
            return
        dx, dy, z = self._random_target(n)
        self.target_pos_world[envs_idx, 0] = self.base_pos[envs_idx, 0] + dx
        self.target_pos_world[envs_idx, 1] = self.base_pos[envs_idx, 1] + dy
        self.target_pos_world[envs_idx, 2] = z

    # =====================================================================
    # End-effector + observation helpers
    # =====================================================================
    def _update_ee_pos(self) -> None:
        """World-frame end-effector position (flange + local tool offset)."""
        ee_link_pos = self.ee_link.get_pos()
        ee_link_quat = self.ee_link.get_quat()
        if ee_link_pos.ndim == 1:
            ee_link_pos = ee_link_pos.unsqueeze(0)
            ee_link_quat = ee_link_quat.unsqueeze(0)
        offset_world = transform_by_quat(
            self.ee_offset.unsqueeze(0).expand(ee_link_pos.shape[0], -1),
            ee_link_quat,
        )
        self.ee_pos = ee_link_pos + offset_world

    def _recompute_target_rel(self) -> None:
        inv_q = inv_quat(self.base_quat)
        self.ball_rel_body = transform_by_quat(
            self.target_pos_world - self.base_pos, inv_q
        )
        self.ee_to_ball_world = self.target_pos_world - self.ee_pos
        self.ee_to_ball_body = transform_by_quat(self.ee_to_ball_world, inv_q)

    # =====================================================================
    # Core RL interface
    # =====================================================================
    def step(self, actions: torch.Tensor):
        self.actions = torch.clip(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.action_scale + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_q = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_q)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_q)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_q)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self._update_ee_pos()
        self._recompute_target_rel()

        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # Respawn the ball when the EE reaches it or after a timeout.
        self.target_step_counter += 1
        dist_ee = torch.linalg.norm(self.ee_to_ball_world, dim=1)
        reached = dist_ee < self.target_reach_dist
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
        if envs_idx is None:
            n = self.num_envs
            idx_all = torch.arange(n, device=self.device, dtype=gs.tc_int)
            self._apply_reset(idx_all, n)
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
            self.target_step_counter.zero_()

            self.extras["episode"] = {}
            for key, value in self.episode_sums.items():
                self.extras["episode"]["rew_" + key] = (
                    value.mean() / self.env_cfg["episode_length_s"]
                )
                value.zero_()

            self._update_ee_pos()
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

        self.extras["episode"] = {}
        denom = float(n) * self.env_cfg["episode_length_s"]
        for key, value in self.episode_sums.items():
            self.extras["episode"]["rew_" + key] = value[envs_idx].sum() / denom
            value[envs_idx] = 0.0

        self._spawn_targets_at_indices(envs_idx)
        self.target_step_counter[envs_idx] = 0
        self._update_ee_pos()
        self._recompute_target_rel()

    def _apply_reset(self, envs_idx: torch.Tensor, n: int) -> None:
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
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],                 # 3
                self.projected_gravity,                                          # 3
                self.ball_rel_body * self.obs_scales["target_pos"],              # 3
                self.ee_to_ball_body * self.obs_scales["target_pos"],            # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 18
                self.dof_vel * self.obs_scales["dof_vel"],                       # 18
                self.actions,                                                    # 18
            ),
            dim=-1,
        )

    # =====================================================================
    # Reward functions
    # =====================================================================
    def _reward_tracking_ee(self):
        """Bell-shaped reward on the 3-D end-effector -> ball distance."""
        dist_sq = torch.sum(self.ee_to_ball_world ** 2, dim=1)
        sigma = self.reward_cfg.get("ee_sigma", 0.3)
        return self.target_coeff * torch.exp(-dist_sq / sigma)

    def _reward_self_collision(self):
        """Keep-out sphere: penalise the EE entering the trunk volume."""
        d = torch.linalg.norm(self.ee_pos - self.base_pos, dim=1)
        pen = torch.clamp(self.keepout_radius - d, min=0.0)
        return pen ** 2

    def _reward_arm_posture(self):
        """Gentle pull of the arm back to its home pose when not reaching."""
        return torch.sum(
            torch.abs(self.dof_pos[:, self.n_leg :] - self.default_arm_pos), dim=1
        )

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        """Legs only — the arm is free to move for reaching."""
        return torch.sum(
            torch.abs(self.dof_pos[:, : self.n_leg] - self.default_dof_pos[: self.n_leg]),
            dim=1,
        )

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])


# =========================================================================
# Default configuration for B2 + Z1 whole-body control
# =========================================================================
def default_cfgs():
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    arm_joint_names = [f"z1_joint{i}" for i in range(1, 7)]

    default_joint_angles = {
        # Legs — B2 standing pose.
        "FR_hip_joint": 0.0, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
        "FL_hip_joint": 0.0, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
        "RR_hip_joint": 0.0, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
        "RL_hip_joint": 0.0, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
        # Arm — folded home pose (matches spawn_b2_z1.py).
        "z1_joint1": 0.0, "z1_joint2": 1.5, "z1_joint3": -1.0,
        "z1_joint4": 0.0, "z1_joint5": 0.0, "z1_joint6": 0.0,
    }

    env_cfg = {
        "num_actions": 18,
        "urdf_path": B2_Z1_URDF_PATH,
        "leg_joint_names": leg_joint_names,
        "arm_joint_names": arm_joint_names,
        "default_joint_angles": default_joint_angles,
        # Stiff legs (heavy B2), lighter arm.
        "leg_kp": 200.0,
        "leg_kd": 5.0,
        "arm_kp": 60.0,
        "arm_kd": 2.0,
        "termination_if_roll_greater_than": 25,   # deg
        "termination_if_pitch_greater_than": 25,
        "base_init_pos": [0.0, 0.0, 0.62],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "action_scale": 0.25,        # legs
        "arm_action_scale": 0.5,     # arm needs a wider sweep
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "dt": 0.02,
    }
    obs_cfg = {
        "num_obs": 66,
        "obs_scales": {
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "target_pos": 1.0,
        },
    }
    reward_cfg = {
        "base_height_target": 0.50,
        "ee_sigma": 0.3,
        "reward_scales": {
            # ---- Stability (Phase 1) ----
            "lin_vel_z": -1.0,
            "ang_vel_xy": -0.05,
            "orientation": -3.0,
            "base_height": -30.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "arm_posture": -0.05,
            "self_collision": -2.0,
            # ---- EE tracking (ramped via target_coeff in Phase 2) ----
            "tracking_ee": 2.5,
        },
    }
    command_cfg = {"num_commands": 0}
    target_cfg = {
        "min_dist": 0.3,
        "max_dist": 1.2,
        "z_min": 0.25,
        "z_max": 0.85,
        "reach_dist": 0.15,
        "resample_time_s": 8.0,
        "ee_link": "z1_link06",
        "ee_offset": (0.08, 0.0, 0.0),
        "keepout_radius": 0.20,
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
