"""Evaluate a FIXED-manipulator delivery-dog policy on the entrance scene.

Loads the velocity-tracking policy trained by ``delivery_dog_train_vel.py``,
drops the entrance (door + keypad) scene into the world as static scenery, and
lets you drive the dog around it with the keyboard. The entrance plays no role
in training — it is purely a backdrop to walk in.

Keys
----
    ↑ / ↓   lin_vel_x  (forward / backward, ramps while held)
    ← / →   lin_vel_y  (left / right)
    Q / E   ang_vel_yaw (ccw / cw)
    SPACE   zero the command
    ESC     quit
All commands are clamped to the training range stored in cfgs.pkl.

Usage::

    uv run src/examples/delivery_dog_eval_vel.py -e dd-walk --ckpt -1
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

from b2_vel_env import B2VelEnv
from delivery_dog_cfgs import ENTRANCE_MJCF, ENTRANCE_SPAWN_POS, ENTRANCE_OFFSET_X

LIN_VEL_RATE_MPS2 = 1.5
ANG_VEL_RATE_RPS2 = 2.0


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
    parser.add_argument("-e", "--exp_name", type=str, default="dd-walk")
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
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    reward_cfg = dict(reward_cfg)
    reward_cfg["reward_scales"] = {}

    # Inject the entrance scene (pushed forward) + spawn the dog near origin.
    if not args.no_entrance:
        env_cfg = dict(env_cfg)
        env_cfg["extra_mjcf"] = ENTRANCE_MJCF
        offset_x = ENTRANCE_OFFSET_X if args.entrance_x is None else args.entrance_x
        env_cfg["extra_mjcf_pos"] = [offset_x, 0.0, 0.0]
        env_cfg["base_init_pos"] = list(ENTRANCE_SPAWN_POS)

    env = B2VelEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    env.enable_external_commands(True)

    ckpt = args.ckpt if args.ckpt >= 0 else _latest_ckpt(log_dir)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoints in {log_dir}")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()

    ranges = env.command_ranges()
    vx_lo, vx_hi = ranges["lin_vel_x_range"]
    vy_lo, vy_hi = ranges["lin_vel_y_range"]
    w_lo, w_hi = ranges["ang_vel_range"]

    cmd = np.zeros(3, dtype=np.float32)

    def _bump_vx(d): cmd[0] = float(np.clip(cmd[0] + d, vx_lo, vx_hi))
    def _bump_vy(d): cmd[1] = float(np.clip(cmd[1] + d, vy_lo, vy_hi))
    def _bump_w(d):  cmd[2] = float(np.clip(cmd[2] + d, w_lo, w_hi))
    def _zero():     cmd[:] = 0.0

    running = {"value": True}
    def _quit(): running["value"] = False

    vx_step = LIN_VEL_RATE_MPS2 * env.dt
    vy_step = LIN_VEL_RATE_MPS2 * env.dt
    w_step = ANG_VEL_RATE_RPS2 * env.dt
    env.scene.viewer.register_keybinds(
        Keybind("dd_fwd",   Key.UP,     KeyAction.HOLD, callback=_bump_vx, args=( vx_step,)),
        Keybind("dd_back",  Key.DOWN,   KeyAction.HOLD, callback=_bump_vx, args=(-vx_step,)),
        Keybind("dd_left",  Key.LEFT,   KeyAction.HOLD, callback=_bump_vy, args=( vy_step,)),
        Keybind("dd_right", Key.RIGHT,  KeyAction.HOLD, callback=_bump_vy, args=(-vy_step,)),
        Keybind("dd_ccw",   Key.Q,      KeyAction.HOLD, callback=_bump_w,  args=( w_step,)),
        Keybind("dd_cw",    Key.E,      KeyAction.HOLD, callback=_bump_w,  args=(-w_step,)),
        Keybind("dd_zero",  Key.SPACE,  KeyAction.PRESS, callback=_zero),
        Keybind("dd_quit",  Key.ESCAPE, KeyAction.RELEASE, callback=_quit),
    )

    print(
        f"[dd-vel] training ranges: lin_x∈[{vx_lo:.2f},{vx_hi:.2f}] "
        f"lin_y∈[{vy_lo:.2f},{vy_hi:.2f}] ang∈[{w_lo:.2f},{w_hi:.2f}]\n"
        "Controls: ↑/↓ vx, ←/→ vy, Q/E yaw, SPACE stop, ESC quit."
    )

    with torch.no_grad():
        while running["value"]:
            env.set_external_commands(cmd.tolist())
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    main()
