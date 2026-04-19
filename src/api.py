"""High-level Python API for the trained Unitree B2 policies.

The :class:`Robot` class exposes a single, intent-level interface that
works for **both** trained policy families:

* ``mode="ball"``     — policy trained by ``b2_train.py`` + ``b2_env.py``.
                        A virtual red ball is placed in the world and the
                        policy walks the robot toward it.

* ``mode="velocity"`` — policy trained by ``b2_train_vel.py`` + ``b2_vel_env.py``.
                        The policy tracks commanded ``(lin_vel_x, lin_vel_y,
                        ang_vel_yaw)``. ``Robot`` adds a small P-controller
                        on top that converts a world-frame target
                        ``(x, y, yaw)`` into body-frame velocity commands
                        clamped to the policy's training range.

From the caller's perspective both modes behave the same:

::

    from api import Robot

    # Point-target policy:
    r = Robot(exp_name="b2-target-rl", mode="ball", show_viewer=True)
    r.move(1.0, 0.0, time=3.0)
    r.rotate(math.pi / 2, time=2.0)
    r.close()

    # Velocity-tracking policy (retrained with wide command ranges):
    r = Robot(exp_name="b2-walk-omni", mode="velocity", show_viewer=True)
    r.move(1.0, 0.0, time=3.0)
    r.rotate(math.pi / 2, time=2.0)
    r.close()

See ``api.md`` for a step-by-step methodology on extending the API (e.g.
when a Z1 manipulator is bolted onto the robot).
"""

from __future__ import annotations

import math
import os
import pickle
import sys
from importlib import metadata

import numpy as np
import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs


_EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)


# =========================================================================
# Helpers
# =========================================================================
def _latest_checkpoint(log_dir: str) -> int | None:
    if not os.path.isdir(log_dir):
        return None
    best = None
    for name in os.listdir(log_dir):
        if name.startswith("model_") and name.endswith(".pt"):
            try:
                it = int(name[len("model_") : -len(".pt")])
            except ValueError:
                continue
            best = it if best is None else max(best, it)
    return best


def _wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# =========================================================================
# Ball-tracking backend (policy commands virtual-target position/yaw)
# =========================================================================
class _BallBackend:
    """Drives ``b2_env.B2TargetEnv`` with an externally-set target ball."""

    def __init__(
        self,
        log_dir: str,
        ckpt_id: int,
        show_viewer: bool,
        draw_target: bool,
    ) -> None:
        from b2_env import B2TargetEnv

        with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
            payload = pickle.load(f)
        if len(payload) != 7:
            raise ValueError(
                "cfgs.pkl for mode='ball' must contain 7 entries produced by "
                f"b2_train.py (got {len(payload)}). Check that --exp_name points "
                "to a ball-policy log, or use mode='velocity' instead."
            )
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, target_cfg, dynamic_cfg = payload

        # Rewards are unused at inference time.
        reward_cfg = dict(reward_cfg)
        reward_cfg["reward_scales"] = {}

        self.env = B2TargetEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            target_cfg=target_cfg,
            show_viewer=show_viewer,
        )
        self.env.enable_external_target(True)

        state_path = os.path.join(log_dir, f"dynamic_state_{ckpt_id}.pkl")
        if os.path.isfile(state_path):
            state = pickle.load(open(state_path, "rb"))
            self.env.target_coeff = state.get(
                "target_coeff", dynamic_cfg.get("max_target_coeff", 1.0)
            )
        else:
            self.env.target_coeff = dynamic_cfg.get("max_target_coeff", 1.0)

        self._runner = OnPolicyRunner(self.env, train_cfg, log_dir, device=gs.device)
        self._runner.load(os.path.join(log_dir, f"model_{ckpt_id}.pt"))
        self._policy = self._runner.get_inference_policy(device=gs.device)

        obs, _ = self.env.reset()
        self._obs = obs

        # Pin the initial target at the robot's starting pose so the robot
        # stands still until set_target() is issued.
        self._target_xy = (
            float(self.env.init_base_pos[0].item()),
            float(self.env.init_base_pos[1].item()),
        )
        self._target_yaw = 0.0
        self.env.set_external_target(pos_xy=self._target_xy, yaw=self._target_yaw)

        self._draw_target = draw_target and show_viewer
        self._target_marker = None

    # -- state ----------------------------------------------------------------
    @property
    def pos(self) -> np.ndarray:
        return self.env.base_pos[0].detach().cpu().numpy()

    @property
    def yaw(self) -> float:
        return float(self.env.base_yaw[0].item())

    @property
    def dt(self) -> float:
        return self.env.dt

    # -- commands -------------------------------------------------------------
    def set_target(
        self,
        x: float | None = None,
        y: float | None = None,
        yaw: float | None = None,
    ) -> None:
        if x is not None:
            self._target_xy = (float(x), self._target_xy[1])
        if y is not None:
            self._target_xy = (self._target_xy[0], float(y))
        if yaw is not None:
            self._target_yaw = float(yaw)
        self.env.set_external_target(pos_xy=self._target_xy, yaw=self._target_yaw)

    def step(self, n: int = 1) -> None:
        with torch.no_grad():
            for _ in range(n):
                if self._draw_target:
                    if self._target_marker is not None:
                        self.env.scene.clear_debug_object(self._target_marker)
                    self._target_marker = self.env.scene.draw_debug_sphere(
                        pos=(
                            self._target_xy[0],
                            self._target_xy[1],
                            float(self.env.target_pos_world[0, 2].item()),
                        ),
                        radius=0.08,
                        color=(1.0, 0.1, 0.1, 0.9),
                    )
                actions = self._policy(self._obs)
                self._obs, _, _, _ = self.env.step(actions)

    def close(self) -> None:
        if self._target_marker is not None:
            try:
                self.env.scene.clear_debug_object(self._target_marker)
            except Exception:
                pass
            self._target_marker = None


# =========================================================================
# Velocity-tracking backend (policy consumes commanded body-frame velocity)
# =========================================================================
class _VelocityBackend:
    """Drives ``b2_vel_env.B2VelEnv`` via a small P-controller on top of the
    velocity-tracking policy.

    A call to :meth:`set_target` ``(x, y, yaw)`` is interpreted as a desired
    pose in the world frame. On every simulation step the backend computes
    the body-frame position error, scales it by :attr:`kp_lin`, and clamps
    to the policy's training range. The result is written through
    ``env.set_external_commands``.
    """

    def __init__(
        self,
        log_dir: str,
        ckpt_id: int,
        show_viewer: bool,
        draw_target: bool,
    ) -> None:
        from b2_vel_env import B2VelEnv

        with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
            payload = pickle.load(f)
        if len(payload) != 5:
            raise ValueError(
                "cfgs.pkl for mode='velocity' must contain 5 entries produced "
                f"by b2_train_vel.py (got {len(payload)}). Check that "
                "--exp_name points to a velocity-policy log, or use mode='ball'."
            )
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = payload

        reward_cfg = dict(reward_cfg)
        reward_cfg["reward_scales"] = {}

        self.env = B2VelEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )
        self.env.enable_external_commands(True)

        self._runner = OnPolicyRunner(self.env, train_cfg, log_dir, device=gs.device)
        self._runner.load(os.path.join(log_dir, f"model_{ckpt_id}.pt"))
        self._policy = self._runner.get_inference_policy(device=gs.device)

        ranges = self.env.command_ranges()
        self.vx_lo, self.vx_hi = ranges["lin_vel_x_range"]
        self.vy_lo, self.vy_hi = ranges["lin_vel_y_range"]
        self.w_lo, self.w_hi = ranges["ang_vel_range"]

        obs, _ = self.env.reset()
        self._obs = obs

        # P-controller gains (mapping position/yaw error → command).
        self.kp_lin = 2.0       # m/s per metre of body-frame error
        self.kp_ang = 2.0       # rad/s per rad of yaw error
        self.pos_tol = 0.10     # dead-zone on XY error (m)
        self.yaw_tol = 0.05     # dead-zone on yaw error (rad)

        self._target_xy = (
            float(self.env.init_base_pos[0].item()),
            float(self.env.init_base_pos[1].item()),
        )
        self._target_yaw = 0.0
        # When non-None, bypasses the P-controller and forwards the stored
        # (vx, vy, ω) tuple straight to the policy. Written by
        # :meth:`set_raw_command`, cleared by :meth:`set_target`.
        self._raw_command: tuple[float, float, float] | None = None

        self._draw_target = draw_target and show_viewer
        self._target_marker = None
        self._target_z = float(self.env.init_base_pos[2].item())

        # Issue a zero command so the robot stands still until set_target()
        # is issued explicitly.
        self.env.set_external_commands((
            float(np.clip(0.0, self.vx_lo, self.vx_hi)),
            float(np.clip(0.0, self.vy_lo, self.vy_hi)),
            float(np.clip(0.0, self.w_lo, self.w_hi)),
        ))

    # -- state ----------------------------------------------------------------
    @property
    def pos(self) -> np.ndarray:
        return self.env.base_pos[0].detach().cpu().numpy()

    @property
    def yaw(self) -> float:
        return float(self.env.base_yaw[0].item())

    @property
    def dt(self) -> float:
        return self.env.dt

    # -- commands -------------------------------------------------------------
    def set_target(
        self,
        x: float | None = None,
        y: float | None = None,
        yaw: float | None = None,
    ) -> None:
        # Switching back to pose-target control disables any prior raw override.
        self._raw_command = None
        if x is not None:
            self._target_xy = (float(x), self._target_xy[1])
        if y is not None:
            self._target_xy = (self._target_xy[0], float(y))
        if yaw is not None:
            self._target_yaw = float(yaw)

    def set_raw_command(self, vx: float, vy: float, w: float) -> None:
        """Bypass the pose-target P-controller and send a fixed body-frame
        velocity command ``(vx, vy, ω)`` on every subsequent step, clamped to
        the policy's training range.
        """
        self._raw_command = (
            float(np.clip(vx, self.vx_lo, self.vx_hi)),
            float(np.clip(vy, self.vy_lo, self.vy_hi)),
            float(np.clip(w, self.w_lo, self.w_hi)),
        )

    def _compute_command(self) -> tuple[float, float, float]:
        if self._raw_command is not None:
            return self._raw_command
        rx = float(self.env.base_pos[0, 0].item())
        ry = float(self.env.base_pos[0, 1].item())
        ryaw = float(self.env.base_yaw[0].item())

        world_dx = self._target_xy[0] - rx
        world_dy = self._target_xy[1] - ry
        dist = math.hypot(world_dx, world_dy)

        if dist < self.pos_tol:
            body_dx = body_dy = 0.0
        else:
            cos_t, sin_t = math.cos(ryaw), math.sin(ryaw)
            body_dx = cos_t * world_dx + sin_t * world_dy
            body_dy = -sin_t * world_dx + cos_t * world_dy

        vx_cmd = float(np.clip(body_dx * self.kp_lin, self.vx_lo, self.vx_hi))
        vy_cmd = float(np.clip(body_dy * self.kp_lin, self.vy_lo, self.vy_hi))

        yaw_err = _wrap_angle(self._target_yaw - ryaw)
        if abs(yaw_err) < self.yaw_tol:
            yaw_err = 0.0
        w_cmd = float(np.clip(yaw_err * self.kp_ang, self.w_lo, self.w_hi))

        return vx_cmd, vy_cmd, w_cmd

    def step(self, n: int = 1) -> None:
        with torch.no_grad():
            for _ in range(n):
                cmd = self._compute_command()
                self.env.set_external_commands(cmd)
                if self._draw_target:
                    if self._target_marker is not None:
                        self.env.scene.clear_debug_object(self._target_marker)
                    self._target_marker = self.env.scene.draw_debug_sphere(
                        pos=(self._target_xy[0], self._target_xy[1], self._target_z),
                        radius=0.08,
                        color=(1.0, 0.1, 0.1, 0.9),
                    )
                actions = self._policy(self._obs)
                self._obs, _, _, _ = self.env.step(actions)

    def close(self) -> None:
        if self._target_marker is not None:
            try:
                self.env.scene.clear_debug_object(self._target_marker)
            except Exception:
                pass
            self._target_marker = None


# =========================================================================
# Public facade
# =========================================================================
class Robot:
    """Unified, mode-agnostic handle onto a trained B2 policy.

    Parameters
    ----------
    exp_name:
        Experiment folder under ``log_root`` produced by the matching train
        script.
    ckpt:
        Iteration id to load. ``-1`` picks the latest ``model_*.pt`` inside
        ``log_root/exp_name``.
    mode:
        ``"ball"`` or ``"velocity"`` — selects which env/backend to drive.
    show_viewer:
        Whether to open the Genesis viewer window.
    log_root:
        Root folder where policy logs live (default ``"logs"``).
    backend:
        Genesis backend: ``"cpu"`` (safe for laptops) or ``"gpu"``.
    draw_target:
        When the viewer is on, draw a red debug sphere at the current
        virtual target for visual feedback.
    """

    def __init__(
        self,
        exp_name: str,
        ckpt: int = -1,
        mode: str = "ball",
        show_viewer: bool = True,
        log_root: str = "logs",
        backend: str = "cpu",
        draw_target: bool = True,
    ) -> None:
        if mode not in ("ball", "velocity"):
            raise ValueError(f"Unknown mode={mode!r}. Use 'ball' or 'velocity'.")

        log_dir = os.path.join(log_root, exp_name)
        if not os.path.isfile(os.path.join(log_dir, "cfgs.pkl")):
            train_script = (
                "src/examples/b2_train.py" if mode == "ball"
                else "src/examples/b2_train_vel.py"
            )
            raise FileNotFoundError(
                f"Cannot find cfgs.pkl under {log_dir}. Train a policy first "
                f"with `uv run {train_script}`."
            )

        gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu)

        ckpt_id = ckpt if ckpt >= 0 else _latest_checkpoint(log_dir)
        if ckpt_id is None:
            raise FileNotFoundError(f"No checkpoints found in {log_dir}.")

        self.mode = mode
        if mode == "ball":
            self._backend = _BallBackend(log_dir, ckpt_id, show_viewer, draw_target)
        else:
            self._backend = _VelocityBackend(log_dir, ckpt_id, show_viewer, draw_target)

    # ===================================================================
    # State
    # ===================================================================
    @property
    def pos(self) -> np.ndarray:
        """World-frame base position ``[x, y, z]`` as a ``(3,)`` array."""
        return self._backend.pos

    @property
    def yaw(self) -> float:
        """Base yaw in the world frame, radians in ``(-pi, pi]``."""
        return self._backend.yaw

    @property
    def dt(self) -> float:
        """Simulation timestep, seconds."""
        return self._backend.dt

    @property
    def env(self):
        """Underlying Genesis env (``B2TargetEnv`` or ``B2VelEnv``)."""
        return self._backend.env

    # ===================================================================
    # Low-level control
    # ===================================================================
    def set_target(
        self,
        x: float | None = None,
        y: float | None = None,
        yaw: float | None = None,
    ) -> None:
        """Update the virtual world-frame target. ``None`` keeps the
        previous value for that axis.

        In ``"ball"`` mode this writes directly to the env's target.
        In ``"velocity"`` mode the backend continuously re-derives the
        velocity command from this target every step.
        """
        self._backend.set_target(x=x, y=y, yaw=yaw)

    def step(self, n: int = 1) -> None:
        """Advance the policy + simulation by ``n`` substeps."""
        self._backend.step(n)

    def set_velocity(self, vx: float, vy: float, w: float) -> None:
        """Send a constant body-frame velocity command ``(vx, vy, ω)``.

        Only available in ``mode="velocity"``. Values are clipped to the
        policy's training range (read from ``cfgs.pkl``). The override holds
        until the next call to :meth:`set_target`, :meth:`move` or
        :meth:`rotate`, which cancels it.
        """
        if self.mode != "velocity":
            raise RuntimeError(
                "set_velocity() is only available in mode='velocity'. "
                "For the ball policy, use set_target()/move()/rotate() instead."
            )
        self._backend.set_raw_command(vx, vy, w)

    # ===================================================================
    # High-level intents (identical across modes)
    # ===================================================================
    def move(self, x: float, y: float, time: float) -> None:
        """Walk toward world point ``(x, y)`` for ``time`` seconds.

        The target yaw is locked to the robot's current heading so the robot
        walks straight rather than pirouetting. The exact trajectory is the
        policy's responsibility; ``time`` is a wall-clock duration, not a
        guarantee of arrival.
        """
        current_yaw = self.yaw
        self.set_target(x=x, y=y, yaw=current_yaw)
        self._run_for(time)

    def rotate(self, yaw: float, time: float) -> None:
        """Yaw to world-frame heading ``yaw`` for ``time`` seconds without
        translating. The target XY is pinned to the robot's current
        position, so the robot rotates in place.
        """
        robot_xy = self.pos[:2].tolist()
        self.set_target(x=robot_xy[0], y=robot_xy[1], yaw=yaw)
        self._run_for(time)

    # ===================================================================
    # Utilities
    # ===================================================================
    def close(self) -> None:
        self._backend.close()

    # ===================================================================
    # Private helpers
    # ===================================================================
    def _run_for(self, time_s: float) -> None:
        n_steps = max(1, int(round(time_s / self.dt)))
        self.step(n_steps)


__all__ = ["Robot"]


if __name__ == "__main__":
    # Demo: drive the robot on a circle for 10 seconds at constant body-frame
    # velocity. Radius of the traced arc is roughly `LIN_VEL_X / ANG_VEL_YAW`.
    #
    # Reproduces the checkpoint used in:
    #   uv run src/examples/b2_eval.py --mode velocity -e b2-walk-omni --ckpt 1200
    EXP_NAME = "b2-walk-omni"
    CKPT = 1200
    DURATION_S = 10.0
    LIN_VEL_X = 0.5        # m/s forward
    LIN_VEL_Y = 0.0        # m/s lateral
    ANG_VEL_YAW = 0.5      # rad/s → radius ≈ 1.0 m

    robot = Robot(
        exp_name=EXP_NAME,
        ckpt=CKPT,
        mode="velocity",
        show_viewer=True,
        backend="cpu",
        draw_target=False,
    )
    try:
        robot.set_velocity(vx=LIN_VEL_X, vy=LIN_VEL_Y, w=ANG_VEL_YAW)
        n_steps = max(1, int(round(DURATION_S / robot.dt)))
        print(
            f"[api demo] driving circle: vx={LIN_VEL_X}, vy={LIN_VEL_Y}, "
            f"w={ANG_VEL_YAW}, steps={n_steps}, dt={robot.dt:.4f}s"
        )
        robot.step(n_steps)
        print(f"[api demo] final pos={robot.pos}, yaw={robot.yaw:.3f}")
    finally:
        robot.close()
