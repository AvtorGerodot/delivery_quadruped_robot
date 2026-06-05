"""Spawn the combined Unitree B2 + Z1 robot in Genesis to verify the
mount geometry before training.

The quadruped legs are held at the B2 standing pose and the Z1 arm at a
configurable, folded "home" pose — both with position PD control — so the
robot stands still and you can inspect that the manipulator sits correctly
on the spine and nothing self-collides badly.

Build the URDF first (once, or after changing the mount)::

    python complex_urdf/build_b2_z1.py

Then spawn::

    uv run src/examples/spawn_b2_z1.py                 # with viewer
    uv run src/examples/spawn_b2_z1.py --seconds 5 --no-viewer   # headless check

Tune the arm rest pose with ``--arm-home j1 j2 j3 j4 j5 j6`` (radians).
"""

from __future__ import annotations

import argparse
import os

import torch

import genesis as gs


COMBINED_URDF = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "complex_urdf", "b2_z1.urdf")
)

# B2 standing pose — identical to b2_env.default_cfgs().
LEG_JOINTS = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
LEG_HOME = {
    "FR_hip_joint": 0.0, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
    "FL_hip_joint": 0.0, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
    "RR_hip_joint": 0.0, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
    "RL_hip_joint": 0.0, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
}
LEG_KP, LEG_KD = 200.0, 5.0

# Z1 arm joints (renamed with z1_ prefix in the merged URDF).
ARM_JOINTS = [f"z1_joint{i}" for i in range(1, 7)]
# Folded rest pose: lift link02 up (joint2) and fold the forearm back
# (joint3) so the arm sits compactly over the back instead of sticking out.
ARM_HOME_DEFAULT = [0.0, 1.5, -1.0, 0.0, 0.0, 0.0]
ARM_KP, ARM_KD = 80.0, 2.0

BASE_INIT_POS = (0.0, 0.0, 0.62)
BASE_INIT_QUAT = (1.0, 0.0, 0.0, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spawn the B2+Z1 combined robot")
    parser.add_argument("--urdf", default=COMBINED_URDF, help="Path to the merged URDF.")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--no-viewer", action="store_true", help="Run headless.")
    parser.add_argument("--seconds", type=float, default=1e9,
                        help="Sim seconds to run (default: until window closed).")
    parser.add_argument("--arm-home", type=float, nargs=6, default=ARM_HOME_DEFAULT,
                        metavar=tuple(f"j{i}" for i in range(1, 7)),
                        help="Z1 arm home pose, radians.")
    parser.add_argument("--merge-fixed-links", action="store_true", default=True,
                        help="Merge fixed links (default on, matches b2_env).")
    args = parser.parse_args()

    if not os.path.isfile(args.urdf):
        raise SystemExit(
            f"URDF not found: {args.urdf}\n"
            "Build it first with:  python complex_urdf/build_b2_z1.py"
        )

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.02, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 1.6, 1.4),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
            max_FPS=50,
        ),
        show_viewer=not args.no_viewer,
    )

    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=args.urdf,
            pos=BASE_INIT_POS,
            quat=BASE_INIT_QUAT,
            merge_fixed_links=args.merge_fixed_links,
        ),
    )

    scene.build(n_envs=1)

    # ── Diagnostics: list links, joints, DOFs ─────────────────────────
    print("\n" + "=" * 64)
    print(f"  Spawned: {args.urdf}")
    print("=" * 64)
    link_names = [ln.name for ln in robot.links]
    print(f"  Links ({len(link_names)}):")
    print("   ", ", ".join(link_names))
    arm_links = [n for n in link_names if n.startswith("z1_")]
    print(f"  Z1 links on the body: {arm_links}")

    # Resolve DOF indices for leg + arm joints (skip any merged away).
    def dof_of(name: str):
        try:
            return robot.get_joint(name).dof_start
        except Exception:
            return None

    leg_dofs, leg_names_ok = [], []
    for n in LEG_JOINTS:
        d = dof_of(n)
        if d is not None:
            leg_dofs.append(d)
            leg_names_ok.append(n)
    arm_dofs, arm_names_ok = [], []
    for n in ARM_JOINTS:
        d = dof_of(n)
        if d is not None:
            arm_dofs.append(d)
            arm_names_ok.append(n)

    print(f"\n  Leg DOFs ({len(leg_dofs)}): {dict(zip(leg_names_ok, leg_dofs))}")
    print(f"  Arm DOFs ({len(arm_dofs)}): {dict(zip(arm_names_ok, arm_dofs))}")
    print(f"  Total robot DOFs: {robot.n_dofs}")

    dev = gs.device
    leg_dof_idx = torch.tensor(leg_dofs, dtype=gs.tc_int, device=dev)
    arm_dof_idx = (
        torch.tensor(arm_dofs, dtype=gs.tc_int, device=dev) if arm_dofs else None
    )

    # ── Gains ─────────────────────────────────────────────────────────
    robot.set_dofs_kp([LEG_KP] * len(leg_dofs), leg_dof_idx)
    robot.set_dofs_kv([LEG_KD] * len(leg_dofs), leg_dof_idx)
    if arm_dof_idx is not None:
        robot.set_dofs_kp([ARM_KP] * len(arm_dofs), arm_dof_idx)
        robot.set_dofs_kv([ARM_KD] * len(arm_dofs), arm_dof_idx)

    # ── Seat the home pose ────────────────────────────────────────────
    leg_target = torch.tensor(
        [LEG_HOME[n] for n in leg_names_ok], dtype=gs.tc_float, device=dev
    ).unsqueeze(0)
    robot.set_dofs_position(position=leg_target, dofs_idx_local=leg_dof_idx,
                            zero_velocity=True)
    if arm_dof_idx is not None:
        arm_home = torch.tensor(
            [args.arm_home[int(n[-1]) - 1] for n in arm_names_ok],
            dtype=gs.tc_float, device=dev,
        ).unsqueeze(0)
        robot.set_dofs_position(position=arm_home, dofs_idx_local=arm_dof_idx,
                                zero_velocity=True)
        print(f"  Arm home pose: {dict(zip(arm_names_ok, args.arm_home))}")

    # ── Hold pose under gravity ───────────────────────────────────────
    n_steps = int(args.seconds / 0.02) if args.seconds < 1e8 else 10_000_000
    print("\n  Holding pose. Close the viewer window or Ctrl-C to stop.\n")
    try:
        for step in range(n_steps):
            robot.control_dofs_position(leg_target, leg_dof_idx)
            if arm_dof_idx is not None:
                robot.control_dofs_position(arm_home, arm_dof_idx)
            scene.step()
            if step % 100 == 0:
                base_z = float(robot.get_pos()[0, 2].item()) if robot.get_pos().ndim == 2 \
                    else float(robot.get_pos()[2].item())
                print(f"    step={step:>5}  base_z={base_z:.3f} m")
    except KeyboardInterrupt:
        print("\n  Interrupted.")


if __name__ == "__main__":
    main()
