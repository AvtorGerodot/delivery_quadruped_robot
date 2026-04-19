"""Evaluate a trained B2 target-tracking policy with arrow-key control.

Loads the latest (or user-specified) checkpoint produced by
``b2_train.py`` and drops into an interactive Genesis viewer. The
red virtual ball is steered from the keyboard arrow keys — the policy
chases it.

Controls
--------
    UP   / DOWN     ball moves forward / backward in robot body frame
    LEFT / RIGHT    ball moves left / right in robot body frame
    Q / E           target yaw rotates counter-clockwise / clockwise
    R               ball snapped back to the robot's current pose
    SPACE           zero the ball offset and target yaw (robot stops)
    ESC             quit

Usage::

    uv run src/examples/b2_eval.py -e b2-target-rl --ckpt 499
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

from b2_env import B2TargetEnv


# Body-frame speed at which held keys drag the virtual ball (m/s, rad/s).
BALL_SPEED_MS = 1.2
YAW_SPEED_RADS = 1.5


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="b2-target-rl")
    parser.add_argument(
        "--ckpt",
        type=int,
        default=-1,
        help="Iteration id to load (default: latest found in logs/<exp_name>/).",
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, target_cfg, dynamic_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    # Disable reward computation during evaluation (the masked-fill reset
    # path requires a non-empty episode_sums dict to be valid).
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

    ckpt = args.ckpt if args.ckpt >= 0 else _latest_checkpoint(log_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoints found in {log_dir}")

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

    # Body-frame offset of the ball, controlled by the arrow keys.
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
        "Controls: ↑/↓/←/→ move target (body frame), Q/E rotate target yaw, "
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


if __name__ == "__main__":
    main()
