"""Train the delivery-dog with WHOLE-BODY CONTROL (legs + arm reach a ball).

Same task and adaptive reward scheduling as ``b2_z1_wbc_train.py`` but on the
customer's delivery-dog URDF: the policy controls 12 legs + 6 arm joints to put
the gripper finger tip on a world-frame ball; the gripper joint is held static.

Usage::

    uv run src/examples/delivery_dog_wbc_train.py -e dd-wbc \\
        --backend gpu --device cuda -B 4096 --max_iterations 2000

Evaluate on the entrance scene afterwards::

    uv run src/examples/delivery_dog_wbc_eval.py -e dd-wbc --ckpt -1
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil

import torch

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from b2_z1_wbc_env import B2Z1WholeBodyEnv
from b2_z1_wbc_train import get_train_cfg, dynamic_learn
from delivery_dog_cfgs import wbc_cfgs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="dd-wbc")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=2000)
    parser.add_argument("--backend", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    env_cfg, obs_cfg, reward_cfg, command_cfg, target_cfg, dynamic_cfg = wbc_cfgs(
        entrance=False
    )
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, target_cfg, dynamic_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, precision="32", logging_level="warning",
            seed=train_cfg["seed"], performance_mode=True)

    if args.device is None:
        policy_device = str(gs.device)
    else:
        policy_device = args.device
        if policy_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"--device={policy_device} requested but torch.cuda.is_available() is False."
            )

    print(
        f"[delivery_dog_wbc_train] backend={args.backend} device={policy_device} "
        f"num_envs={args.num_envs}\n"
        f"  urdf={env_cfg['urdf_path']}  actions={env_cfg['num_actions']} "
        f"(12 legs + 6 arm)\n"
        f"  static joint (gripper): {env_cfg['static_joint_names']}"
    )

    env = B2Z1WholeBodyEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        target_cfg=target_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=policy_device)
    dynamic_learn(runner, env, args.max_iterations, dynamic_cfg, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
