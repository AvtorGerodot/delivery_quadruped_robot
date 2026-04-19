"""Evaluate a trained B2 policy with arrow-key control.

Two modes, selected via ``--mode``:

* ``--mode ball``      — policy from ``b2_train.py`` + ``b2_env.py``.
                          The red ball is steered in the robot body frame;
                          the policy chases it.

* ``--mode velocity``  — policy from ``b2_train_vel.py`` + ``b2_vel_env.py``.
                          Arrows directly set ``(lin_vel_x, lin_vel_y)``,
                          Q / E set ``ang_vel_yaw``. All commands are
                          **clamped to the training range** read from
                          ``cfgs.pkl`` so the policy stays inside its
                          training distribution.

Common keys
-----------
    SPACE   zero the command / stop the robot
    ESC     quit

Ball-mode keys
--------------
    ↑ / ↓   ball moves forward / backward (body frame)
    ← / →   ball moves left / right       (body frame)
    Q / E   target yaw rotates ccw / cw
    R       snap ball back to the robot's current pose

Velocity-mode keys
------------------
    ↑ / ↓   increase / decrease lin_vel_x   (held = ramp up)
    ← / →   increase lin_vel_y / decrease lin_vel_y
    Q / E   increase / decrease ang_vel_yaw

Usage::

    # Ball-tracking policy:
    uv run src/examples/b2_eval.py --mode ball     -e b2-target-rl --ckpt 499

    # Velocity-tracking policy:
    uv run src/examples/b2_eval.py --mode velocity -e b2-walk      --ckpt 499
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
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
from genesis.vis.keybindings import Key, KeyAction, Keybind


# Body-frame speed at which held keys drag the virtual ball (m/s, rad/s).
BALL_SPEED_MS = 1.2
YAW_SPEED_RADS = 1.5

# Rate at which held arrow keys ramp the velocity command (per-second,
# clamped to the training range downstream).
LIN_VEL_RATE_MPS2 = 1.5   # lin_vel rate of change
ANG_VEL_RATE_RPS2 = 2.0   # ang_vel rate of change


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


def _resolve_ckpt_id(log_dir: str, ckpt: int) -> int:
    if ckpt < 0:
        ckpt = _latest_checkpoint(log_dir)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoints found in {log_dir}")
    return ckpt


# =========================================================================
# Ball-mode eval (unchanged from the previous version, just factored out)
# =========================================================================
def _run_ball(args) -> None:
    from b2_env import B2TargetEnv

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, target_cfg, dynamic_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    reward_cfg = dict(reward_cfg)
    reward_cfg["reward_scales"] = {}

    env = B2TargetEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        target_cfg=target_cfg,
        show_viewer=True,
    )
    env.enable_external_target(True)

    ckpt = _resolve_ckpt_id(log_dir, args.ckpt)
    state_path = os.path.join(log_dir, f"dynamic_state_{ckpt}.pkl")
    if os.path.exists(state_path):
        state = pickle.load(open(state_path, "rb"))
        env.target_coeff = state.get("target_coeff", dynamic_cfg.get("max_target_coeff", 1.0))
        print(f"Loaded dynamic state: target_coeff={env.target_coeff:.3f} phase={state.get('phase')}")
    else:
        env.target_coeff = dynamic_cfg.get("max_target_coeff", 1.0)

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()

    ball_body_offset = np.zeros(2, dtype=np.float32)   # (fwd, left)
    target_yaw_world = np.float32(0.0)
    env.set_external_target(
        pos_xy=(float(env.init_base_pos[0]), float(env.init_base_pos[1])),
        yaw=0.0,
    )

    ball_marker = [None]

    def _add_body(dfwd: float, dlat: float) -> None:
        ball_body_offset[0] = float(np.clip(ball_body_offset[0] + dfwd, -3.0, 3.0))
        ball_body_offset[1] = float(np.clip(ball_body_offset[1] + dlat, -3.0, 3.0))

    def _rotate_target(dyaw: float) -> None:
        nonlocal target_yaw_world
        target_yaw_world = float(
            ((target_yaw_world + dyaw) + math.pi) % (2 * math.pi) - math.pi
        )

    def _zero_offset() -> None:
        ball_body_offset[:] = 0.0

    def _snap_to_robot() -> None:
        nonlocal target_yaw_world
        ball_body_offset[:] = 0.0
        target_yaw_world = float(env.base_yaw[0].item())

    running = {"value": True}

    def _quit() -> None:
        running["value"] = False

    v_step = BALL_SPEED_MS * env.dt
    w_step = YAW_SPEED_RADS * env.dt
    env.scene.viewer.register_keybinds(
        Keybind("b2_fwd",     Key.UP,    KeyAction.HOLD, callback=_add_body, args=( v_step, 0.0)),
        Keybind("b2_back",    Key.DOWN,  KeyAction.HOLD, callback=_add_body, args=(-v_step, 0.0)),
        Keybind("b2_left",    Key.LEFT,  KeyAction.HOLD, callback=_add_body, args=(0.0,  v_step)),
        Keybind("b2_right",   Key.RIGHT, KeyAction.HOLD, callback=_add_body, args=(0.0, -v_step)),
        Keybind("b2_yaw_ccw", Key.Q,     KeyAction.HOLD, callback=_rotate_target, args=( w_step,)),
        Keybind("b2_yaw_cw",  Key.E,     KeyAction.HOLD, callback=_rotate_target, args=(-w_step,)),
        Keybind("b2_zero",    Key.SPACE, KeyAction.PRESS, callback=_zero_offset),
        Keybind("b2_reset",   Key.R,     KeyAction.PRESS, callback=_snap_to_robot),
        Keybind("b2_quit",    Key.ESCAPE, KeyAction.RELEASE, callback=_quit),
    )

    print(
        "[ball] Controls: ↑/↓/←/→ move target (body frame), Q/E rotate target yaw, "
        "SPACE zero offset, R snap ball to robot, ESC quit."
    )

    with torch.no_grad():
        while running["value"]:
            robot_pos = env.base_pos[0].cpu().numpy()
            robot_yaw = float(env.base_yaw[0].item())
            cos_y, sin_y = math.cos(robot_yaw), math.sin(robot_yaw)
            world_dx = cos_y * ball_body_offset[0] - sin_y * ball_body_offset[1]
            world_dy = sin_y * ball_body_offset[0] + cos_y * ball_body_offset[1]
            tgt_x = float(robot_pos[0] + world_dx)
            tgt_y = float(robot_pos[1] + world_dy)
            env.set_external_target(pos_xy=(tgt_x, tgt_y), yaw=target_yaw_world)

            if ball_marker[0] is not None:
                env.scene.clear_debug_object(ball_marker[0])
            ball_marker[0] = env.scene.draw_debug_sphere(
                pos=(tgt_x, tgt_y, float(env.target_pos_world[0, 2].item())),
                radius=0.08,
                color=(1.0, 0.1, 0.1, 0.9),
            )

            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


# =========================================================================
# Velocity-mode eval
# =========================================================================
def _run_velocity(args) -> None:
    from b2_vel_env import B2VelEnv

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    reward_cfg = dict(reward_cfg)
    reward_cfg["reward_scales"] = {}

    env = B2VelEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    env.enable_external_commands(True)

    ckpt = _resolve_ckpt_id(log_dir, args.ckpt)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()

    # Command limits from the training config so the eval always stays
    # inside the policy's training distribution.
    ranges = env.command_ranges()
    vx_lo, vx_hi = ranges["lin_vel_x_range"]
    vy_lo, vy_hi = ranges["lin_vel_y_range"]
    w_lo, w_hi = ranges["ang_vel_range"]

    # Start at zero, clamped into the training support.
    cmd = np.array(
        [
            float(np.clip(0.0, vx_lo, vx_hi)),
            float(np.clip(0.0, vy_lo, vy_hi)),
            float(np.clip(0.0, w_lo, w_hi)),
        ],
        dtype=np.float32,
    )
    env.set_external_commands(cmd.tolist())

    def _bump_vx(d: float) -> None:
        cmd[0] = float(np.clip(cmd[0] + d, vx_lo, vx_hi))

    def _bump_vy(d: float) -> None:
        cmd[1] = float(np.clip(cmd[1] + d, vy_lo, vy_hi))

    def _bump_w(d: float) -> None:
        cmd[2] = float(np.clip(cmd[2] + d, w_lo, w_hi))

    def _zero_cmd() -> None:
        cmd[0] = float(np.clip(0.0, vx_lo, vx_hi))
        cmd[1] = float(np.clip(0.0, vy_lo, vy_hi))
        cmd[2] = float(np.clip(0.0, w_lo, w_hi))

    running = {"value": True}

    def _quit() -> None:
        running["value"] = False

    vx_step = LIN_VEL_RATE_MPS2 * env.dt
    vy_step = LIN_VEL_RATE_MPS2 * env.dt
    w_step = ANG_VEL_RATE_RPS2 * env.dt
    env.scene.viewer.register_keybinds(
        Keybind("b2v_fwd",     Key.UP,    KeyAction.HOLD, callback=_bump_vx, args=( vx_step,)),
        Keybind("b2v_back",    Key.DOWN,  KeyAction.HOLD, callback=_bump_vx, args=(-vx_step,)),
        # Left arrow = +lin_vel_y (leftward in body frame, ROS convention).
        Keybind("b2v_left",    Key.LEFT,  KeyAction.HOLD, callback=_bump_vy, args=( vy_step,)),
        Keybind("b2v_right",   Key.RIGHT, KeyAction.HOLD, callback=_bump_vy, args=(-vy_step,)),
        Keybind("b2v_yaw_ccw", Key.Q,     KeyAction.HOLD, callback=_bump_w,  args=( w_step,)),
        Keybind("b2v_yaw_cw",  Key.E,     KeyAction.HOLD, callback=_bump_w,  args=(-w_step,)),
        Keybind("b2v_zero",    Key.SPACE, KeyAction.PRESS, callback=_zero_cmd),
        Keybind("b2v_quit",    Key.ESCAPE, KeyAction.RELEASE, callback=_quit),
    )

    print(
        "[velocity] training ranges (commands are clamped to these):\n"
        f"  lin_vel_x ∈ [{vx_lo:.3f}, {vx_hi:.3f}] m/s\n"
        f"  lin_vel_y ∈ [{vy_lo:.3f}, {vy_hi:.3f}] m/s\n"
        f"  ang_vel_z ∈ [{w_lo:.3f}, {w_hi:.3f}] rad/s\n"
        "Controls: ↑/↓ lin_vel_x, ←/→ lin_vel_y, Q/E ang_vel, "
        "SPACE zero, ESC quit."
    )

    degenerate = (vx_lo == vx_hi) and (vy_lo == vy_hi) and (w_lo == w_hi)
    if degenerate:
        print(
            "[velocity] NOTE: training ranges are degenerate (single value) -- "
            "keyboard input will have no effect. Retrain with wider "
            "--lin_vel_x_range / --lin_vel_y_range / --ang_vel_range."
        )

    print_every = max(1, int(0.25 / env.dt))
    step = 0
    with torch.no_grad():
        while running["value"]:
            env.set_external_commands(cmd.tolist())
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            step += 1
            if step % print_every == 0:
                bv = env.base_lin_vel[0].cpu().numpy()
                bw = float(env.base_ang_vel[0, 2].item())
                print(
                    f"  cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})  "
                    f"base_vel=({bv[0]:+.2f}, {bv[1]:+.2f}, {bw:+.2f})",
                    end="\r",
                    flush=True,
                )


# =========================================================================
# Main
# =========================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ball", "velocity"],
        default="ball",
        help="Which kind of policy / env to evaluate.",
    )
    parser.add_argument("-e", "--exp_name", type=str, default="b2-target-rl")
    parser.add_argument(
        "--ckpt",
        type=int,
        default=-1,
        help="Iteration id to load (default: latest found in logs/<exp_name>/).",
    )
    args = parser.parse_args()

    if args.mode == "ball":
        _run_ball(args)
    else:
        _run_velocity(args)


if __name__ == "__main__":
    main()
