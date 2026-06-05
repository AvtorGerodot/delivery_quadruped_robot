"""Evaluate a trained B2 + Z1 whole-body reaching policy with keyboard control.

The red ball lives in the **world** frame and is driven in 3-D. The policy
walks the quadruped and moves the Z1 arm so its end-effector touches the ball.
A small green sphere marks the current end-effector position so you can see the
tracking error.

Keys
----
    ↑ / ↓   ball  +x / -x   (world frame)
    ← / →   ball  +y / -y   (world frame)
    Q / E   ball  up / down (z, clamped to the reachable band)
    R       reset the ball to a reachable point in front of the robot
    SPACE   park the ball at the current end-effector (robot holds still)
    ESC     quit

Usage::

    uv run src/examples/b2_z1_wbc_eval.py -e b2-z1-wbc --ckpt -1
"""

from __future__ import annotations

import argparse
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

from b2_z1_wbc_env import B2Z1WholeBodyEnv


# Speed at which held keys drag the world-frame ball (m/s).
BALL_SPEED_MS = 0.6


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="b2-z1-wbc")
    parser.add_argument(
        "--ckpt",
        type=int,
        default=-1,
        help="Iteration id to load (default: latest found in logs/<exp_name>/).",
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, target_cfg, dynamic_cfg = pickle.load(f)

    reward_cfg = dict(reward_cfg)
    reward_cfg["reward_scales"] = {}

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

    z_min = float(target_cfg.get("z_min", 0.25))
    z_max = float(target_cfg.get("z_max", 0.85))

    def _default_ball() -> np.ndarray:
        base = env.base_pos[0].cpu().numpy()
        return np.array(
            [base[0] + 0.5, base[1], 0.5 * (z_min + z_max)], dtype=np.float32
        )

    ball = _default_ball()
    env.set_external_target(tuple(ball.tolist()))

    def _move(dx: float, dy: float) -> None:
        ball[0] += dx
        ball[1] += dy

    def _move_z(dz: float) -> None:
        ball[2] = float(np.clip(ball[2] + dz, z_min, z_max))

    def _reset_ball() -> None:
        ball[:] = _default_ball()

    def _park_at_ee() -> None:
        ball[:] = env.ee_pos[0].cpu().numpy()

    running = {"value": True}

    def _quit() -> None:
        running["value"] = False

    v = BALL_SPEED_MS * env.dt
    env.scene.viewer.register_keybinds(
        Keybind("wbc_xp",   Key.UP,     KeyAction.HOLD, callback=_move,   args=( v, 0.0)),
        Keybind("wbc_xn",   Key.DOWN,   KeyAction.HOLD, callback=_move,   args=(-v, 0.0)),
        Keybind("wbc_yp",   Key.LEFT,   KeyAction.HOLD, callback=_move,   args=(0.0,  v)),
        Keybind("wbc_yn",   Key.RIGHT,  KeyAction.HOLD, callback=_move,   args=(0.0, -v)),
        Keybind("wbc_zp",   Key.Q,      KeyAction.HOLD, callback=_move_z, args=( v,)),
        Keybind("wbc_zn",   Key.E,      KeyAction.HOLD, callback=_move_z, args=(-v,)),
        Keybind("wbc_reset", Key.R,     KeyAction.PRESS, callback=_reset_ball),
        Keybind("wbc_park", Key.SPACE,  KeyAction.PRESS, callback=_park_at_ee),
        Keybind("wbc_quit", Key.ESCAPE, KeyAction.RELEASE, callback=_quit),
    )

    print(
        f"[wbc] reachable z band: [{z_min:.2f}, {z_max:.2f}] m\n"
        "Controls: ↑/↓ ball ±x, ←/→ ball ±y, Q/E ball ±z, "
        "R reset ball, SPACE park at EE, ESC quit."
    )

    ball_marker = [None]
    ee_marker = [None]
    print_every = max(1, int(0.25 / env.dt))
    step = 0
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
            step += 1
            if step % print_every == 0:
                dist = float(np.linalg.norm(ee - ball))
                print(
                    f"  ball=({ball[0]:+.2f}, {ball[1]:+.2f}, {ball[2]:+.2f})  "
                    f"ee=({ee[0]:+.2f}, {ee[1]:+.2f}, {ee[2]:+.2f})  "
                    f"dist={dist:.3f} m",
                    end="\r",
                    flush=True,
                )


if __name__ == "__main__":
    main()
