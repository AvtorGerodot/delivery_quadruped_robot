"""Keyboard teleoperation of the Unitree B2 quadruped in the Genesis simulator.

The URDF is loaded directly from ``unitree_ros/robots/b2_description`` so the
model matches the one used on the real robot. The locomotion itself is
produced by a simple open-loop trot gait that converts the user's
``(vx, vy, wz)`` body-velocity command into twelve target joint angles, which
Genesis then tracks with a per-joint PD controller.

The controller is intentionally lightweight: it is enough to visually drive
the robot around the scene from the keyboard, while keeping the command
interface (`VelocityCommand`) identical to what the Unitree SDK2
``SportClient.Move(vx, vy, wz)`` call expects. That way the same front-end
can later be pointed at the real robot, or replaced with a trained RL
policy (see ``Genesis/examples/locomotion/go2_train.py`` for a template).

Controls
--------
    W / S         hold to ramp forward / backward velocity
    A / D         hold to ramp strafe left / right velocity
    Q / E         hold to ramp yaw counter-clockwise / clockwise
    SPACE         zero the velocity command (gentle stop)
    R             teleport robot back to the start pose
    ESC           quit

Run with::

    uv run src/example_b2_teleop.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
B2_URDF_PATH = os.path.join(
    PROJECT_ROOT,
    "unitree_ros",
    "robots",
    "b2_description",
    "urdf",
    "b2_description.urdf",
)

# The joint ordering used by Unitree's low-level API and by the RL baselines
# in ``Genesis/examples/locomotion``. Keeping this order makes it trivial to
# later swap the open-loop gait for a trained policy.
JOINT_NAMES: list[str] = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

# Comfortable stand pose. Front legs are less flexed than the rears so that
# the trunk sits roughly level. Values are close to the Unitree GO2/B2
# factory defaults.
DEFAULT_JOINT_ANGLES: dict[str, float] = {
    "FR_hip_joint": 0.0, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
    "FL_hip_joint": 0.0, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
    "RR_hip_joint": 0.0, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
    "RL_hip_joint": 0.0, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
}

# Initial base pose: hold the robot slightly above the ground so that the
# stand pose settles onto the plane rather than penetrating it.
BASE_INIT_POS = np.array([0.0, 0.0, 0.62], dtype=np.float32)
BASE_INIT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Simulation constants.
SIM_DT = 0.005  # 200 Hz physics tick
CONTROL_DECIMATION = 4  # PD setpoints updated at 50 Hz


@dataclass
class VelocityCommand:
    """Body-frame velocity command, identical in shape to SportClient.Move."""

    vx: float = 0.0  # forward, m/s
    vy: float = 0.0  # lateral, m/s (positive = left)
    wz: float = 0.0  # yaw rate, rad/s (positive = counter-clockwise)

    def magnitude(self) -> float:
        return abs(self.vx) + abs(self.vy) + 0.3 * abs(self.wz)

    def clear(self) -> None:
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0


class TrotGaitController:
    """Minimal open-loop trot gait.

    Diagonal leg pairs (FL+RR and FR+RL) swing in anti-phase. The target
    joint angles are composed from:

    * a static stand pose (``default_angles``),
    * a sagittal stride term that modulates the thigh pitch with ``cos(phi)``
      to move the foot forward/backward,
    * a swing-height term that lifts the calf during the swing half-cycle.

    This is deliberately simple -- it is a placeholder for a proper MPC or
    learned policy, and is enough to demonstrate teleoperation.
    """

    def __init__(
        self,
        default_angles: dict[str, float],
        joint_names: list[str],
        stride_freq_hz: float = 2.0,
        swing_lift: float = 0.22,      # rad, thigh lift at the top of swing
        swing_bend: float = 0.45,      # rad, extra calf flexion during swing
        stride_gain: float = 0.35,     # rad of thigh pitch per (m/s forward)
        yaw_gain: float = 0.18,        # rad of thigh pitch per (rad/s yaw)
        lateral_gain: float = 0.30,    # rad of hip roll per (m/s strafe)
        deadband: float = 0.05,
    ) -> None:
        self.default_angles = default_angles
        self.joint_names = joint_names
        self.stride_freq_hz = stride_freq_hz
        self.swing_lift = swing_lift
        self.swing_bend = swing_bend
        self.stride_gain = stride_gain
        self.yaw_gain = yaw_gain
        self.lateral_gain = lateral_gain
        self.deadband = deadband

        self._default_vec = np.array(
            [default_angles[n] for n in joint_names], dtype=np.float32
        )
        # Trot phase offsets: diagonal pairs in anti-phase.
        self._leg_phase = {"FL": 0.0, "RR": 0.0, "FR": np.pi, "RL": np.pi}

    def compute(self, t: float, cmd: VelocityCommand) -> np.ndarray:
        if cmd.magnitude() < self.deadband:
            return self._default_vec.copy()

        phase = 2.0 * np.pi * self.stride_freq_hz * t
        out = self._default_vec.copy()

        for i, name in enumerate(self.joint_names):
            leg = name[:2]
            is_left = leg.endswith("L")
            phi = phase + self._leg_phase[leg]
            sin_phi = float(np.sin(phi))
            cos_phi = float(np.cos(phi))
            lift = max(0.0, sin_phi)  # non-negative on swing half-cycle

            stride = self.stride_gain * cmd.vx
            stride += self.yaw_gain * cmd.wz * (-1.0 if is_left else 1.0)

            if name.endswith("hip_joint"):
                # Push the body sideways by abducting the hips synchronously
                # with the swing; sign flips between left and right legs so
                # both push in the commanded direction.
                sign = 1.0 if is_left else -1.0
                out[i] += sign * self.lateral_gain * cmd.vy * (0.5 + 0.5 * cos_phi)
            elif name.endswith("thigh_joint"):
                out[i] += cos_phi * stride - lift * self.swing_lift
            elif name.endswith("calf_joint"):
                out[i] -= lift * self.swing_bend

        return out


def build_scene() -> tuple[gs.Scene, "gs.engine.entities.RigidEntity"]:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=SIM_DT,
            substeps=2,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            gravity=(0.0, 0.0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 2.0, 1.4),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=45,
            max_FPS=60,
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=B2_URDF_PATH,
            pos=tuple(BASE_INIT_POS.tolist()),
            quat=tuple(BASE_INIT_QUAT.tolist()),
            merge_fixed_links=True,
        ),
    )
    scene.build()
    return scene, robot


def run() -> None:
    gs.init(backend=gs.cpu, logging_level="info")

    scene, robot = build_scene()

    dof_indices = np.array(
        [robot.get_joint(n).dof_start for n in JOINT_NAMES], dtype=np.int64
    )

    # PD gains ordered as (hip, thigh, calf) * 4 legs to match JOINT_NAMES.
    # Tuned loosely from unitree_ros/robots/b2_description/config/robot_control.yaml.
    kp = np.array([100.0, 300.0, 300.0] * 4, dtype=np.float32)
    kd = np.array([  5.0,   8.0,   8.0] * 4, dtype=np.float32)
    robot.set_dofs_kp(kp, dof_indices)
    robot.set_dofs_kv(kd, dof_indices)

    default_qpos = np.array(
        [DEFAULT_JOINT_ANGLES[n] for n in JOINT_NAMES], dtype=np.float32
    )
    robot.set_dofs_position(default_qpos, dof_indices, zero_velocity=True)

    controller = TrotGaitController(DEFAULT_JOINT_ANGLES, JOINT_NAMES)
    cmd = VelocityCommand()

    # How much each HOLD tick changes the command. The Genesis viewer fires
    # HOLD callbacks at the viewer frame-rate (~60 Hz), so 0.01 ≈ 0.6 m/s per
    # second of holding -- a comfortable ramp for a robot this size.
    V_STEP = 0.01
    W_STEP = 0.03
    V_MAX = 1.2
    W_MAX = 1.5

    def _add_vx(dv: float) -> None:
        cmd.vx = float(np.clip(cmd.vx + dv, -V_MAX, V_MAX))

    def _add_vy(dv: float) -> None:
        cmd.vy = float(np.clip(cmd.vy + dv, -V_MAX, V_MAX))

    def _add_wz(dw: float) -> None:
        cmd.wz = float(np.clip(cmd.wz + dw, -W_MAX, W_MAX))

    running = {"value": True}

    def _stop_cmd() -> None:
        cmd.clear()

    def _reset_robot() -> None:
        robot.set_dofs_position(default_qpos, dof_indices, zero_velocity=True)
        robot.set_pos(BASE_INIT_POS)
        robot.set_quat(BASE_INIT_QUAT)
        cmd.clear()

    def _quit() -> None:
        running["value"] = False

    scene.viewer.register_keybinds(
        Keybind("b2_fwd",     Key.W, KeyAction.HOLD, callback=_add_vx, args=( V_STEP,)),
        Keybind("b2_back",    Key.S, KeyAction.HOLD, callback=_add_vx, args=(-V_STEP,)),
        Keybind("b2_left",    Key.A, KeyAction.HOLD, callback=_add_vy, args=( V_STEP,)),
        Keybind("b2_right",   Key.D, KeyAction.HOLD, callback=_add_vy, args=(-V_STEP,)),
        Keybind("b2_yaw_ccw", Key.Q, KeyAction.HOLD, callback=_add_wz, args=( W_STEP,)),
        Keybind("b2_yaw_cw",  Key.E, KeyAction.HOLD, callback=_add_wz, args=(-W_STEP,)),
        Keybind("b2_stop",    Key.SPACE, KeyAction.PRESS, callback=_stop_cmd),
        Keybind("b2_reset",   Key.R, KeyAction.PRESS, callback=_reset_robot),
        Keybind("b2_quit",    Key.ESCAPE, KeyAction.RELEASE, callback=_quit),
    )

    gs.logger.info(
        "Teleop ready. Keys: W/S forward/back, A/D strafe, Q/E yaw, "
        "SPACE stop, R reset, ESC quit."
    )

    t = 0.0
    step = 0
    target = default_qpos.copy()
    try:
        while running["value"]:
            if step % CONTROL_DECIMATION == 0:
                target = controller.compute(t, cmd)
            robot.control_dofs_position(target, dof_indices)
            scene.step()
            t += SIM_DT
            step += 1
    except KeyboardInterrupt:
        gs.logger.info("Interrupted by user.")
    finally:
        gs.logger.info(
            "Final command: vx=%.2f vy=%.2f wz=%.2f" % (cmd.vx, cmd.vy, cmd.wz)
        )


if __name__ == "__main__":
    run()
