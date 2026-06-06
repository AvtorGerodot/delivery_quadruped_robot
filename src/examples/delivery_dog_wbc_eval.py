"""Evaluate a WHOLE-BODY-CONTROL delivery-dog policy on the entrance scene.

Loads the policy trained by ``delivery_dog_wbc_train.py``, drops the entrance
(door + keypad) scene into the world as static scenery, and lets you drive a
3-D world-frame ball that the gripper finger tip tracks. A green sphere marks
the current end-effector. The entrance plays no role in training.

Keys
----
    ↑ / ↓   ball  +x / -x   (world frame)
    ← / →   ball  +y / -y   (world frame)
    Q / E   ball  up / down (z, clamped to the reachable band)
    R       reset the ball in front of the robot
    SPACE   park the ball at the current end-effector
    ESC     quit

Usage::

    uv run src/examples/delivery_dog_wbc_eval.py -e dd-wbc --ckpt -1
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind

from b2_z1_wbc_env import B2Z1WholeBodyEnv
from delivery_dog_cfgs import ENTRANCE_MJCF, ENTRANCE_SPAWN_POS, ENTRANCE_OFFSET_X

BALL_SPEED_MS = 0.6


def _latest_ckpt(log_dir: str):
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
    parser.add_argument("-e", "--exp_name", type=str, default="dd-wbc")
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--no-entrance", action="store_true",
                        help="Eval on a bare plane instead of the entrance scene.")
    parser.add_argument("--entrance-x", type=float, default=None,
                        help="Forward (+x) offset of the entrance scene in metres "
                             "(default from delivery_dog_cfgs).")
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, target_cfg, dynamic_cfg = pickle.load(f)

    reward_cfg = dict(reward_cfg)
    reward_cfg["reward_scales"] = {}

    if not args.no_entrance:
        env_cfg = dict(env_cfg)
        env_cfg["extra_mjcf"] = ENTRANCE_MJCF
        offset_x = ENTRANCE_OFFSET_X if args.entrance_x is None else args.entrance_x
        env_cfg["extra_mjcf_pos"] = [offset_x, 0.0, 0.0]
        env_cfg["base_init_pos"] = list(ENTRANCE_SPAWN_POS)

    env = B2Z1WholeBodyEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        target_cfg=target_cfg,
        show_viewer=True,
    )
    env.enable_external_target(True)

    ckpt = args.ckpt if args.ckpt >= 0 else _latest_ckpt(log_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoints in {log_dir}")
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

    z_min = float(target_cfg.get("z_min", 0.3))
    z_max = float(target_cfg.get("z_max", 1.0))

    def _default_ball():
        base = env.base_pos[0].cpu().numpy()
        return np.array([base[0] + 0.5, base[1], 0.5 * (z_min + z_max)], dtype=np.float32)

    ball = _default_ball()
    env.set_external_target(tuple(ball.tolist()))

    def _move(dx, dy):
        ball[0] += dx
        ball[1] += dy
    def _move_z(dz):
        ball[2] = float(np.clip(ball[2] + dz, z_min, z_max))
    def _reset_ball():
        ball[:] = _default_ball()
    def _park():
        ball[:] = env.ee_pos[0].cpu().numpy()

    running = {"value": True}
    def _quit(): running["value"] = False

    v = BALL_SPEED_MS * env.dt
    env.scene.viewer.register_keybinds(
        Keybind("ddw_xp",   Key.UP,     KeyAction.HOLD, callback=_move,   args=( v, 0.0)),
        Keybind("ddw_xn",   Key.DOWN,   KeyAction.HOLD, callback=_move,   args=(-v, 0.0)),
        Keybind("ddw_yp",   Key.LEFT,   KeyAction.HOLD, callback=_move,   args=(0.0,  v)),
        Keybind("ddw_yn",   Key.RIGHT,  KeyAction.HOLD, callback=_move,   args=(0.0, -v)),
        Keybind("ddw_zp",   Key.Q,      KeyAction.HOLD, callback=_move_z, args=( v,)),
        Keybind("ddw_zn",   Key.E,      KeyAction.HOLD, callback=_move_z, args=(-v,)),
        Keybind("ddw_reset", Key.R,     KeyAction.PRESS, callback=_reset_ball),
        Keybind("ddw_park", Key.SPACE,  KeyAction.PRESS, callback=_park),
        Keybind("ddw_quit", Key.ESCAPE, KeyAction.RELEASE, callback=_quit),
    )

    print(
        f"[dd-wbc] reachable z band: [{z_min:.2f}, {z_max:.2f}] m\n"
        "Controls: ↑/↓ ball ±x, ←/→ ball ±y, Q/E ball ±z, R reset, SPACE park, ESC quit."
    )

    ball_marker = [None]
    ee_marker = [None]
    with torch.no_grad():
        while running["value"]:
            env.set_external_target(tuple(ball.tolist()))
            if ball_marker[0] is not None:
                env.scene.clear_debug_object(ball_marker[0])
            ball_marker[0] = env.scene.draw_debug_sphere(
                pos=tuple(ball.tolist()), radius=0.06, color=(1.0, 0.1, 0.1, 0.9)
            )
            ee = env.ee_pos[0].cpu().numpy()
            if ee_marker[0] is not None:
                env.scene.clear_debug_object(ee_marker[0])
            ee_marker[0] = env.scene.draw_debug_sphere(
                pos=tuple(ee.tolist()), radius=0.04, color=(0.1, 1.0, 0.1, 0.9)
            )
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    main()
