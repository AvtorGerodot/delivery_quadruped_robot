"""Drive the trained B2 policy from a DualShock 4 gamepad.

Mapping (follows the request: left stick → virtual ball, right stick X →
yaw, right stick Y → ignored):

    Left stick  X  →  target ball offset to the right   (body frame, m)
    Left stick  Y  →  target ball offset forward        (body frame, m)
    Right stick X  →  yaw rate (rad/s)
    Right stick Y  →  ignored
    Circle / B     →  snap target back to robot pose (stop)
    Options / Start→  quit

Usage::

    uv run src/examples/ds4_control.py -e b2-target-rl

Requires a checkpoint produced by ``b2_train.py`` under ``logs/<exp_name>``.

On Linux the DS4 is picked up automatically if ``hid-playstation`` (kernel
≥ 5.12) or ``ds4drv`` is running. Wired USB and Bluetooth both work.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import pygame

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from api import Robot  # noqa: E402


# ------------------------- Tunables ------------------------------------
MAX_BODY_OFFSET_M = 1.2     # how far ahead of the robot the ball can sit
MAX_YAW_RATE_RADS = 1.8     # rotation speed at full R-stick deflection
DEADZONE = 0.12             # stick noise floor


# ------------------------- DS4 wrapper ---------------------------------
class DS4:
    """Thin wrapper around :mod:`pygame.joystick` tuned for a DualShock 4."""

    AXIS_LX = 0
    AXIS_LY = 1
    AXIS_RX = 2
    # AXIS_RY = 3

    # DualShock 4 mapping via SDL GameController: Circle=1, Square=2,
    # Triangle=3, Options=9 on most Linux builds; fall back gracefully if a
    # particular index does not exist.
    BTN_RESET = 1
    BTN_QUIT = 9

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
        print(f"[DS4] connected: {self._js.get_name()}  "
              f"axes={self._js.get_numaxes()}  buttons={self._js.get_numbuttons()}")

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


# ------------------------- Main loop -----------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="b2-target-rl")
    parser.add_argument("--ckpt", type=int, default=-1)
    args = parser.parse_args()

    ds4 = DS4()

    robot = Robot(exp_name=args.exp_name, ckpt=args.ckpt, show_viewer=True)
    target_yaw = robot.yaw

    try:
        while True:
            ds4.pump()
            if ds4.quit_pressed:
                print("[DS4] quit pressed, exiting.")
                break
            if ds4.reset_pressed:
                target_yaw = robot.yaw
                robot.set_target(x=robot.pos[0], y=robot.pos[1], yaw=target_yaw)
                robot.step(1)
                continue

            lx, ly = ds4.left_stick
            rx = ds4.right_stick_x

            # Left stick → body-frame target offset.
            # Stick up (ly = -1) = robot walks forward; stick right (lx = +1) = walks right.
            fwd = -ly * MAX_BODY_OFFSET_M
            left = -lx * MAX_BODY_OFFSET_M

            # Right stick X → yaw rate; stick right (+rx) = clockwise (negative yaw).
            target_yaw = target_yaw + (-rx * MAX_YAW_RATE_RADS) * robot.dt
            target_yaw = math.atan2(math.sin(target_yaw), math.cos(target_yaw))

            rp = robot.pos
            rtheta = robot.yaw
            cos_t, sin_t = math.cos(rtheta), math.sin(rtheta)
            world_dx = cos_t * fwd - sin_t * left
            world_dy = sin_t * fwd + cos_t * left

            robot.set_target(
                x=float(rp[0] + world_dx),
                y=float(rp[1] + world_dy),
                yaw=target_yaw,
            )
            robot.step(1)

    except KeyboardInterrupt:
        print("[DS4] interrupted, exiting.")
    finally:
        robot.close()
        pygame.quit()


if __name__ == "__main__":
    main()
