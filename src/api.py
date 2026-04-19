"""Programmatic robot API backed by the trained B2 RL policy.

Provides a thin wrapper around :class:`B2TargetEnv` + a trained policy,
exposing a very small, intent-level interface for scripts, demos, and
teleoperation front-ends (keyboard, DS4 gamepad, etc.).

Example
-------
::

    from api import Robot

    robot = Robot(exp_name="b2-target-rl", show_viewer=True)
    robot.move(1.0, 0.0, time=3.0)     # walk to world point (1, 0)
    robot.rotate(math.pi / 2, time=2.0) # yaw to +90° in place
    robot.close()

Both ``move`` and ``rotate`` spin the simulation synchronously for the
requested duration. For continuous teleoperation (e.g. a gamepad), use the
lower-level :meth:`Robot.set_target` / :meth:`Robot.step` pair instead.
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

from b2_env import B2TargetEnv  # noqa: E402


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


class Robot:
    """High-level, policy-driven interface to the simulated B1 quadruped."""

    def __init__(
        self,
        exp_name: str = "b2-target-rl",
        ckpt: int = -1,
        show_viewer: bool = True,
        log_root: str = "logs",
        backend: str = "cpu",
        draw_target: bool = True,
    ) -> None:
        log_dir = os.path.join(log_root, exp_name)
        if not os.path.isfile(os.path.join(log_dir, "cfgs.pkl")):
            raise FileNotFoundError(
                f"Cannot find cfgs.pkl under {log_dir}. Train a policy first with "
                "`uv run src/examples/b2_train.py`."
            )
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, target_cfg, dynamic_cfg = pickle.load(
            open(os.path.join(log_dir, "cfgs.pkl"), "rb")
        )

        gs.init(backend=gs.gpu if backend == "gpu" else gs.cpu)

        # Disable rewards at inference time - set/masked_fill_ shortcuts in
        # reset() are not None-safe when reward_scales is empty.
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

        ckpt_id = ckpt if ckpt >= 0 else _latest_checkpoint(log_dir)
        if ckpt_id is None:
            raise FileNotFoundError(f"No checkpoints found in {log_dir}.")

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
        # sits still until the user issues a command.
        self._target_xy = (
            float(self.env.init_base_pos[0].item()),
            float(self.env.init_base_pos[1].item()),
        )
        self._target_yaw = 0.0
        self.env.set_external_target(pos_xy=self._target_xy, yaw=self._target_yaw)

        self._draw_target = draw_target and show_viewer
        self._target_marker = None

    # ---------------------------------------------------------------
    # Low-level control
    # ---------------------------------------------------------------
    def set_target(
        self,
        x: float | None = None,
        y: float | None = None,
        yaw: float | None = None,
    ) -> None:
        """Update the virtual target. ``None`` keeps the current value."""
        if x is not None:
            self._target_xy = (float(x), self._target_xy[1])
        if y is not None:
            self._target_xy = (self._target_xy[0], float(y))
        if yaw is not None:
            self._target_yaw = float(yaw)
        self.env.set_external_target(pos_xy=self._target_xy, yaw=self._target_yaw)

    def step(self, n: int = 1) -> None:
        """Advance the policy + simulation by ``n`` substeps."""
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

    # ---------------------------------------------------------------
    # High-level intents
    # ---------------------------------------------------------------
    def move(self, x: float, y: float, time: float) -> None:
        """Walk the robot toward the virtual ball at world point ``(x, y)``.

        ``time`` is the wall-clock duration of the policy rollout in seconds.
        The target yaw is locked to the robot's current heading so the dog
        walks in a straight line (or curves as needed) rather than pirouetting
        on its way over.
        """
        current_yaw = float(self.env.base_yaw[0].item())
        self.set_target(x=x, y=y, yaw=current_yaw)
        self._run_for(time)

    def rotate(self, yaw: float, time: float) -> None:
        """Yaw the robot to world-frame heading ``yaw`` without translating.

        The target position is pinned to the robot's current position; the
        policy is therefore rewarded only for the yaw alignment term and
        should rotate in place.
        """
        robot_xy = self.env.base_pos[0, :2].tolist()
        self.set_target(x=robot_xy[0], y=robot_xy[1], yaw=yaw)
        self._run_for(time)

    # ---------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------
    @property
    def pos(self) -> np.ndarray:
        return self.env.base_pos[0].detach().cpu().numpy()

    @property
    def yaw(self) -> float:
        return float(self.env.base_yaw[0].item())

    @property
    def dt(self) -> float:
        return self.env.dt

    def close(self) -> None:
        if self._target_marker is not None:
            try:
                self.env.scene.clear_debug_object(self._target_marker)
            except Exception:
                pass
            self._target_marker = None

    # ---------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------
    def _run_for(self, time_s: float) -> None:
        n_steps = max(1, int(round(time_s / self.env.dt)))
        self.step(n_steps)


__all__ = ["Robot"]
