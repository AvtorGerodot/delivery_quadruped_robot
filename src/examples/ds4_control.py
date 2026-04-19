"""Drive a trained B2 policy from a DualShock 4 gamepad.

Two control modes are supported, selected with ``--mode``:

* ``--mode ball``      — the policy trained by ``b2_train.py`` + ``b2_env.py``.
                          Left stick deflects a virtual "red ball" in the
                          robot body frame, right-stick X rotates the target
                          yaw, and the policy chases it.

* ``--mode velocity``  — the policy trained by ``b2_train_vel.py`` +
                          ``b2_vel_env.py``. Left stick directly sets the
                          commanded (lin_vel_x, lin_vel_y), right-stick X
                          sets the commanded yaw rate. All commands are
                          **clamped to the training ranges** read back from
                          ``cfgs.pkl`` so the policy stays inside its
                          training distribution.

Common controls
---------------
    Left stick  X  →  lateral (body frame)
    Left stick  Y  →  forward (body frame; up = forward)
    Right stick X  →  yaw / yaw rate
    Right stick Y  →  ignored
    Circle / B     →  stop the robot (zero command / snap target to robot)
    Options / Start→  quit

Usage::

    # Ball-tracking policy (existing behaviour):
    uv run src/examples/ds4_control.py --mode ball --exp_name b2-target-rl

    # Velocity-tracking policy:
    uv run src/examples/ds4_control.py --mode velocity --exp_name b2-walk

    # Pick a specific checkpoint by iteration id:
    uv run src/examples/ds4_control.py --mode velocity -e b2-walk --ckpt 400

On Linux the DS4 is picked up automatically if ``hid-playstation`` (kernel
>= 5.12) or ``ds4drv`` is running. Both USB and Bluetooth work.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from importlib import metadata

import numpy as np
import pygame
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for p in (_THIS_DIR, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError(
        "Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'."
    ) from e

import genesis as gs
from rsl_rl.runners import OnPolicyRunner


# ------------------------- Tunables ------------------------------------
MAX_BODY_OFFSET_M = 1.2     # ball mode: how far ahead of the robot the ball can sit
MAX_YAW_RATE_RADS = 1.8     # ball mode: rotation speed at full R-stick deflection
DEADZONE = 0.12             # stick noise floor


# =======================================================================
# DS4 gamepad wrapper
# =======================================================================
class DS4:
    """Thin wrapper around :mod:`pygame.joystick` tuned for a DualShock 4."""

    AXIS_LX = 0
    AXIS_LY = 1
    AXIS_RX = 2
    # AXIS_RY = 3

    BTN_RESET = 1   # Circle
    BTN_QUIT = 9    # Options

    def __init__(self) -> None:
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No gamepad detected. Plug in a DualShock 4 (USB or Bluetooth) "
                "and make sure the `hid-playstation` kernel driver or `ds4drv` is loaded."
            )
        self._js = pygame.joystick.Joystick(0)
        self._js.init()
        print(
            f"[DS4] connected: {self._js.get_name()}  "
            f"axes={self._js.get_numaxes()}  buttons={self._js.get_numbuttons()}"
        )

    def _axis(self, idx: int) -> float:
        if idx >= self._js.get_numaxes():
            return 0.0
        v = self._js.get_axis(idx)
        return 0.0 if abs(v) < DEADZONE else float(v)

    def _button(self, idx: int) -> bool:
        if idx >= self._js.get_numbuttons():
            return False
        return bool(self._js.get_button(idx))

    def pump(self) -> None:
        pygame.event.pump()

    @property
    def left_stick(self) -> tuple[float, float]:
        return self._axis(self.AXIS_LX), self._axis(self.AXIS_LY)

    @property
    def right_stick_x(self) -> float:
        return self._axis(self.AXIS_RX)

    @property
    def reset_pressed(self) -> bool:
        return self._button(self.BTN_RESET)

    @property
    def quit_pressed(self) -> bool:
        return self._button(self.BTN_QUIT)


# =======================================================================
# Helpers
# =======================================================================
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


def _resolve_ckpt(log_dir: str, ckpt: str | int) -> str:
    """Resolve --ckpt into an absolute .pt path.

    Accepts an integer iteration id (then ``logs/<exp>/model_<id>.pt``) or a
    filesystem path to a ``.pt`` file.
    """
    if isinstance(ckpt, str) and (ckpt.endswith(".pt") or os.sep in ckpt):
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt}")
        return os.path.abspath(ckpt)
    try:
        ckpt_id = int(ckpt)
    except (TypeError, ValueError):
        raise ValueError(f"Could not interpret --ckpt={ckpt!r} as int or path.")
    if ckpt_id < 0:
        latest = _latest_checkpoint(log_dir)
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found in {log_dir}.")
        ckpt_id = latest
    path = os.path.join(log_dir, f"model_{ckpt_id}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return path


# =======================================================================
# Ball-tracking backend (wraps src/api.py::Robot)
# =======================================================================
class BallBackend:
    def __init__(self, exp_name: str, ckpt: str | int, show_viewer: bool = True):
        from api import Robot  # deferred so the velocity mode doesn't need it

        self.robot = Robot(
            exp_name=exp_name, ckpt=int(ckpt) if str(ckpt).lstrip("-").isdigit() else -1,
            show_viewer=show_viewer,
        )
        self.target_yaw = self.robot.yaw

    def apply_stick(self, lx: float, ly: float, rx: float) -> None:
        # Left stick → body-frame ball offset. Stick up (ly=-1) = forward.
        fwd = -ly * MAX_BODY_OFFSET_M
        left = -lx * MAX_BODY_OFFSET_M

        # Right stick X → yaw rate. Stick right (+rx) = clockwise (negative yaw).
        self.target_yaw = self.target_yaw + (-rx * MAX_YAW_RATE_RADS) * self.robot.dt
        self.target_yaw = math.atan2(math.sin(self.target_yaw), math.cos(self.target_yaw))

        rp = self.robot.pos
        rtheta = self.robot.yaw
        cos_t, sin_t = math.cos(rtheta), math.sin(rtheta)
        world_dx = cos_t * fwd - sin_t * left
        world_dy = sin_t * fwd + cos_t * left

        self.robot.set_target(
            x=float(rp[0] + world_dx),
            y=float(rp[1] + world_dy),
            yaw=self.target_yaw,
        )

    def stop(self) -> None:
        self.target_yaw = self.robot.yaw
        self.robot.set_target(
            x=float(self.robot.pos[0]),
            y=float(self.robot.pos[1]),
            yaw=self.target_yaw,
        )

    def step(self) -> None:
        self.robot.step(1)

    def close(self) -> None:
        self.robot.close()

    def banner(self) -> str:
        return (
            "[ball] Left stick → virtual ball offset (body frame); "
            "Right stick X → target yaw rate."
        )


# =======================================================================
# Velocity-tracking backend (uses b2_vel_env directly)
# =======================================================================
class VelocityBackend:
    def __init__(
        self,
        exp_name: str,
        ckpt: str | int,
        show_viewer: bool = True,
        log_root: str = "logs",
    ) -> None:
        log_dir = os.path.join(log_root, exp_name)
        cfg_path = os.path.join(log_dir, "cfgs.pkl")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f"No cfgs.pkl under {log_dir}. Train first with "
                "`uv run src/examples/b2_train_vel.py`."
            )
        with open(cfg_path, "rb") as f:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

        # Rewards are unused at inference; keep the dict empty so reset() does
        # not touch per-reward buffers.
        reward_cfg = dict(reward_cfg)
        reward_cfg["reward_scales"] = {}

        gs.init(backend=gs.cpu)

        from b2_vel_env import B2VelEnv

        self.env = B2VelEnv(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )
        self.env.enable_external_commands(True)

        ckpt_path = _resolve_ckpt(log_dir, ckpt)
        self._runner = OnPolicyRunner(self.env, train_cfg, log_dir, device=gs.device)
        self._runner.load(ckpt_path)
        self._policy = self._runner.get_inference_policy(device=gs.device)

        obs, _ = self.env.reset()
        self._obs = obs

        ranges = self.env.command_ranges()
        self.vx_lo, self.vx_hi = ranges["lin_vel_x_range"]
        self.vy_lo, self.vy_hi = ranges["lin_vel_y_range"]
        self.w_lo, self.w_hi = ranges["ang_vel_range"]

        # Start at zero command, clamped into the training support.
        self._cmd = np.array(
            [
                float(np.clip(0.0, self.vx_lo, self.vx_hi)),
                float(np.clip(0.0, self.vy_lo, self.vy_hi)),
                float(np.clip(0.0, self.w_lo, self.w_hi)),
            ],
            dtype=np.float32,
        )
        self.env.set_external_commands(self._cmd.tolist())

    @staticmethod
    def _stick_to_range(val: float, lo: float, hi: float) -> float:
        """Map [-1, 1] stick input to [lo, hi]."""
        if hi == lo:
            return float(lo)
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        out = mid + val * half
        return float(np.clip(out, lo, hi))

    def apply_stick(self, lx: float, ly: float, rx: float) -> None:
        # Stick up (ly = -1) = forward (+x). Stick right (+lx) = rightward (-y body).
        vx = self._stick_to_range(-ly, self.vx_lo, self.vx_hi)
        vy = self._stick_to_range(-lx, self.vy_lo, self.vy_hi)
        w = self._stick_to_range(-rx, self.w_lo, self.w_hi)
        self._cmd[:] = (vx, vy, w)
        self.env.set_external_commands(self._cmd.tolist())

    def stop(self) -> None:
        self._cmd[:] = (
            float(np.clip(0.0, self.vx_lo, self.vx_hi)),
            float(np.clip(0.0, self.vy_lo, self.vy_hi)),
            float(np.clip(0.0, self.w_lo, self.w_hi)),
        )
        self.env.set_external_commands(self._cmd.tolist())

    def step(self) -> None:
        with torch.no_grad():
            actions = self._policy(self._obs)
            self._obs, _, _, _ = self.env.step(actions)

    def close(self) -> None:
        pass

    def banner(self) -> str:
        return (
            f"[velocity] command ranges (clamped at stick): "
            f"lin_x={[self.vx_lo, self.vx_hi]}  "
            f"lin_y={[self.vy_lo, self.vy_hi]}  "
            f"ang={[self.w_lo, self.w_hi]}\n"
            f"  Left stick Y → lin_vel_x, Left stick X → lin_vel_y, "
            f"Right stick X → ang_vel_yaw."
        )


# =======================================================================
# Main loop
# =======================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ball", "velocity"],
        required=True,
        help="Control mode. 'ball' = virtual-target policy (b2_env.py); "
             "'velocity' = command-velocity policy (b2_vel_env.py).",
    )
    parser.add_argument(
        "-e", "--exp_name", type=str, required=True,
        help="Experiment folder under logs/ produced by the matching train script.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="-1",
        help="Checkpoint iteration id (e.g. '400'), or a full path to a .pt file. "
             "Default picks the latest model_*.pt in logs/<exp_name>.",
    )
    parser.add_argument("--no_viewer", action="store_true")
    args = parser.parse_args()

    ds4 = DS4()

    if args.mode == "ball":
        backend = BallBackend(
            exp_name=args.exp_name, ckpt=args.ckpt, show_viewer=not args.no_viewer
        )
    else:
        backend = VelocityBackend(
            exp_name=args.exp_name, ckpt=args.ckpt, show_viewer=not args.no_viewer
        )

    print(backend.banner())
    print("Options / Start = quit,  Circle / B = stop.")

    try:
        while True:
            ds4.pump()
            if ds4.quit_pressed:
                print("[DS4] quit pressed, exiting.")
                break
            if ds4.reset_pressed:
                backend.stop()
                backend.step()
                continue

            lx, ly = ds4.left_stick
            rx = ds4.right_stick_x
            backend.apply_stick(lx, ly, rx)
            backend.step()

    except KeyboardInterrupt:
        print("[DS4] interrupted, exiting.")
    finally:
        backend.close()
        pygame.quit()


if __name__ == "__main__":
    main()
