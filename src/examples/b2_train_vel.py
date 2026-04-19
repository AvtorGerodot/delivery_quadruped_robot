"""Train a B2 velocity-tracking policy, go2-style.

Straightforward PPO training loop — no dynamic reward scheduling — that
pairs ``B2VelEnv`` with ``rsl_rl.runners.OnPolicyRunner``. Sampling ranges
for the ``(lin_vel_x, lin_vel_y, ang_vel)`` command are taken from
``default_cfgs()`` (identical to the original go2_train defaults) but can
be widened on the command line via ``--lin_vel_x_range`` etc.

Usage::

    # Default go2-style command ranges (forward walk only):
    uv run src/examples/b2_train_vel.py -e b2-walk --backend gpu --device cuda \\
        -B 4096 --max_iterations 500

    # Wider ranges so the policy learns to walk in every direction:
    uv run src/examples/b2_train_vel.py -e b2-walk-omni \\
        --lin_vel_x_range=-1.0 1.0 --lin_vel_y_range=-0.5 0.5 \\
        --ang_vel_range=-1.0 1.0 -B 4096 --max_iterations 800
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
from importlib import metadata

import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError(
        "Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'."
    ) from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from b2_vel_env import B2VelEnv, default_cfgs


def get_train_cfg(exp_name: str, max_iterations: int) -> dict:
    """PPO runner config compatible with rsl-rl-lib 2.2.4."""
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }


def _range_type(values):
    if len(values) != 2:
        raise argparse.ArgumentTypeError("expected exactly two numbers: LOW HIGH")
    low, high = float(values[0]), float(values[1])
    if low > high:
        raise argparse.ArgumentTypeError(f"LOW ({low}) > HIGH ({high})")
    return [low, high]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="b2-walk")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--backend",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
        help="Genesis backend. Use 'cpu' for smoke-tests on machines without CUDA.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Torch device for the RL policy (e.g. 'cuda', 'cuda:0', 'cpu'). "
            "Defaults to Genesis' own device."
        ),
    )
    parser.add_argument(
        "--lin_vel_x_range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Override [low, high] range for commanded forward velocity (m/s).",
    )
    parser.add_argument(
        "--lin_vel_y_range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Override [low, high] range for commanded lateral velocity (m/s).",
    )
    parser.add_argument(
        "--ang_vel_range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=None,
        help="Override [low, high] range for commanded yaw rate (rad/s).",
    )
    args = parser.parse_args()

    env_cfg, obs_cfg, reward_cfg, command_cfg = default_cfgs()
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
    gs.init(
        backend=backend,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=True,
    )

    if args.device is None:
        policy_device = str(gs.device)
    else:
        policy_device = args.device
        if policy_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"--device={policy_device} requested but torch.cuda.is_available() is False."
            )
    env_device_str = str(gs.device)
    if policy_device.startswith("cuda") and not env_device_str.startswith("cuda"):
        print(
            f"[b2_train_vel] WARNING: policy on {policy_device} but Genesis "
            f"runs on {env_device_str}. Prefer `--backend gpu --device cuda`."
        )
    print(
        f"[b2_train_vel] backend={args.backend} env_device={env_device_str} "
        f"policy_device={policy_device} num_envs={args.num_envs}\n"
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
