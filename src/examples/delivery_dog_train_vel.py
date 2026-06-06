"""Train the delivery-dog with a FIXED manipulator (velocity locomotion).

The Z1 arm + gripper are PD-locked at their home pose; the policy only learns
to walk the quadruped, tracking a commanded ``(lin_vel_x, lin_vel_y,
ang_vel_yaw)``. This is the delivery-dog counterpart of ``b2_train_vel.py`` and
reuses the same ``B2VelEnv`` and PPO runner config.

Usage::

    uv run src/examples/delivery_dog_train_vel.py -e dd-walk \\
        --backend gpu --device cuda \\
        --lin_vel_x_range -1.0 1.0 --lin_vel_y_range -0.5 0.5 \\
        --ang_vel_range -1.0 1.0 -B 4096 --max_iterations 3000

Evaluate on the entrance scene afterwards::

    uv run src/examples/delivery_dog_eval_vel.py -e dd-walk --ckpt -1
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil

import torch

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from b2_vel_env import B2VelEnv
from b2_train_vel import get_train_cfg, _range_type
from delivery_dog_cfgs import velocity_cfgs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="dd-walk")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backend", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lin_vel_x_range", nargs=2, type=float,
                        metavar=("LOW", "HIGH"), default=None)
    parser.add_argument("--lin_vel_y_range", nargs=2, type=float,
                        metavar=("LOW", "HIGH"), default=None)
    parser.add_argument("--ang_vel_range", nargs=2, type=float,
                        metavar=("LOW", "HIGH"), default=None)
    args = parser.parse_args()

    env_cfg, obs_cfg, reward_cfg, command_cfg = velocity_cfgs(entrance=False)
    if args.lin_vel_x_range is not None:
        command_cfg["lin_vel_x_range"] = _range_type(args.lin_vel_x_range)
    if args.lin_vel_y_range is not None:
        command_cfg["lin_vel_y_range"] = _range_type(args.lin_vel_y_range)
    if args.ang_vel_range is not None:
        command_cfg["ang_vel_range"] = _range_type(args.ang_vel_range)

    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    train_cfg["seed"] = args.seed

    log_dir = f"logs/{args.exp_name}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend, precision="32", logging_level="warning",
            seed=args.seed, performance_mode=True)

    if args.device is None:
        policy_device = str(gs.device)
    else:
        policy_device = args.device
        if policy_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"--device={policy_device} requested but torch.cuda.is_available() is False."
            )

    print(
        f"[delivery_dog_train_vel] backend={args.backend} device={policy_device} "
        f"num_envs={args.num_envs}\n"
        f"  urdf={env_cfg['urdf_path']}\n"
        f"  static joints (arm+gripper): {env_cfg['arm_joint_names']}\n"
        f"  command ranges: lin_x={command_cfg['lin_vel_x_range']} "
        f"lin_y={command_cfg['lin_vel_y_range']} ang={command_cfg['ang_vel_range']}"
    )

    env = B2VelEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=policy_device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
