"""Spawn the delivery-dog (B2 + Z1 + gripper + delivery box) in Genesis.

Loads the *cleaned* URDF produced by ``complex_urdf/build_delivery_dog.py``
from the customer's ``delivery_dog_ws`` preview model, seats it at the B2
standing pose with the arm folded, and holds it under gravity so you can
verify the mount geometry, the delivery box, and that nothing explodes.

Optionally drops the **entrance scene** (``entrance_group.xml``) into the world
so you can check the robot stands correctly next to the door / keypad.

Build the URDF first (once)::

    python complex_urdf/build_delivery_dog.py

Then spawn::

    uv run src/examples/spawn_delivery_dog.py                  # robot only
    uv run src/examples/spawn_delivery_dog.py --entrance       # with entrance
    uv run src/examples/spawn_delivery_dog.py --seconds 5 --no-viewer
"""

from __future__ import annotations

import argparse
import os

import torch

import genesis as gs


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
COMBINED_URDF = os.path.join(REPO, "complex_urdf", "delivery_dog_b2_z1.urdf")
ENTRANCE_MJCF = os.path.join(
    REPO,
    "delivery_dog_ws",
    "src",
    "mujoco_models",
    "mjcf",
    "entrance_group.xml",
)

LEG_JOINTS = [
    "b2_FR_hip_joint", "b2_FR_thigh_joint", "b2_FR_calf_joint",
    "b2_FL_hip_joint", "b2_FL_thigh_joint", "b2_FL_calf_joint",
    "b2_RR_hip_joint", "b2_RR_thigh_joint", "b2_RR_calf_joint",
    "b2_RL_hip_joint", "b2_RL_thigh_joint", "b2_RL_calf_joint",
]
LEG_HOME = {
    "b2_FR_hip_joint": 0.0, "b2_FR_thigh_joint": 0.8, "b2_FR_calf_joint": -1.5,
    "b2_FL_hip_joint": 0.0, "b2_FL_thigh_joint": 0.8, "b2_FL_calf_joint": -1.5,
    "b2_RR_hip_joint": 0.0, "b2_RR_thigh_joint": 1.0, "b2_RR_calf_joint": -1.5,
    "b2_RL_hip_joint": 0.0, "b2_RL_thigh_joint": 1.0, "b2_RL_calf_joint": -1.5,
}
LEG_KP, LEG_KD = 200.0, 5.0

ARM_JOINTS = [f"z1_joint{i}" for i in range(1, 7)]
ARM_HOME_DEFAULT = [0.0, 1.5, -1.0, 0.0, 0.0, 0.0]
GRIPPER_JOINT = "z1_jointGripper"
GRIPPER_HOME = 0.0
ARM_KP, ARM_KD = 60.0, 2.0

BASE_INIT_POS = (0.0, 0.0, 0.62)
BASE_INIT_QUAT = (1.0, 0.0, 0.0, 0.0)

# Push the entrance forward (+x) so the dog stands in front of the door
# instead of inside the doorway.
ENTRANCE_OFFSET = (1.6, 0.0, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spawn the delivery-dog robot")
    parser.add_argument("--urdf", default=COMBINED_URDF)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--seconds", type=float, default=1e9)
    parser.add_argument("--entrance", action="store_true",
                        help="Also load the entrance_group scene.")
    parser.add_argument("--entrance-x", type=float, default=ENTRANCE_OFFSET[0],
                        help="Forward (+x) offset of the entrance scene in metres.")
    parser.add_argument("--arm-home", type=float, nargs=6, default=ARM_HOME_DEFAULT)
    args = parser.parse_args()

    if not os.path.isfile(args.urdf):
        raise SystemExit(
            f"URDF not found: {args.urdf}\n"
            "Build it first with:  python complex_urdf/build_delivery_dog.py"
        )

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 2.0, 1.8),
            camera_lookat=(0.0, 0.0, 0.6),
            camera_fov=40,
            max_FPS=50,
        ),
        show_viewer=not args.no_viewer,
    )

    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    # Robot must be the FIRST articulated entity so its dof_start == 0 and the
    # joint indices line up with control_dofs_position(dofs_idx_local=...).
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=args.urdf,
            pos=BASE_INIT_POS,
            quat=BASE_INIT_QUAT,
            merge_fixed_links=True,
        ),
    )
    # The entrance is just scenery — add it AFTER the robot so it doesn't shift
    # the robot's dof indices.
    if args.entrance:
        scene.add_entity(
            gs.morphs.MJCF(
                file=ENTRANCE_MJCF,
                pos=(args.entrance_x, ENTRANCE_OFFSET[1], ENTRANCE_OFFSET[2]),
            )
        )
    scene.build(n_envs=1)

    link_names = [ln.name for ln in robot.links]
    print("\n" + "=" * 64)
    print(f"  Spawned: {args.urdf}")
    print(f"  Links ({len(link_names)}): {link_names}")
    print(f"  Total robot DOFs: {robot.n_dofs}")
    print("=" * 64)

    def dof_of(name):
        try:
            return robot.get_joint(name).dof_start
        except Exception:
            return None

    leg_dofs = [dof_of(n) for n in LEG_JOINTS]
    arm_dofs = [dof_of(n) for n in ARM_JOINTS]
    grip_dof = dof_of(GRIPPER_JOINT)
    print(f"  Leg DOFs: {dict(zip(LEG_JOINTS, leg_dofs))}")
    print(f"  Arm DOFs: {dict(zip(ARM_JOINTS, arm_dofs))}")
    print(f"  Gripper DOF: {grip_dof}")

    dev = gs.device
    leg_idx = torch.tensor(leg_dofs, dtype=gs.tc_int, device=dev)
    arm_idx = torch.tensor(arm_dofs, dtype=gs.tc_int, device=dev)
    grip_idx = torch.tensor([grip_dof], dtype=gs.tc_int, device=dev)

    robot.set_dofs_kp([LEG_KP] * len(leg_dofs), leg_idx)
    robot.set_dofs_kv([LEG_KD] * len(leg_dofs), leg_idx)
    robot.set_dofs_kp([ARM_KP] * len(arm_dofs), arm_idx)
    robot.set_dofs_kv([ARM_KD] * len(arm_dofs), arm_idx)
    robot.set_dofs_kp([ARM_KP], grip_idx)
    robot.set_dofs_kv([ARM_KD], grip_idx)

    leg_target = torch.tensor(
        [LEG_HOME[n] for n in LEG_JOINTS], dtype=gs.tc_float, device=dev
    ).unsqueeze(0)
    arm_target = torch.tensor(args.arm_home, dtype=gs.tc_float, device=dev).unsqueeze(0)
    grip_target = torch.tensor([GRIPPER_HOME], dtype=gs.tc_float, device=dev).unsqueeze(0)

    robot.set_dofs_position(leg_target, leg_idx, zero_velocity=True)
    robot.set_dofs_position(arm_target, arm_idx, zero_velocity=True)
    robot.set_dofs_position(grip_target, grip_idx, zero_velocity=True)

    n_steps = int(args.seconds / 0.02) if args.seconds < 1e8 else 10_000_000
    try:
        for step in range(n_steps):
            robot.control_dofs_position(leg_target, leg_idx)
            robot.control_dofs_position(arm_target, arm_idx)
            robot.control_dofs_position(grip_target, grip_idx)
            scene.step()
            if step % 25 == 0:
                pos = robot.get_pos()
                quat = robot.get_quat()
                base_z = float(pos[0, 2].item()) if pos.ndim == 2 else float(pos[2].item())
                q = quat[0] if quat.ndim == 2 else quat
                print(f"    step={step:>5}  base_z={base_z:.3f} m  quat=({float(q[0]):.2f},{float(q[1]):.2f},{float(q[2]):.2f},{float(q[3]):.2f})")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
