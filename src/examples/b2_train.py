"""Train a B2 target-tracking policy with adaptive reward scheduling.

Structure follows ``Genesis/examples/locomotion/dynamic_ics_dog_train.py``:

Phase 1 — Stability
    ``target_coeff = 0``. Only stability rewards (base height, orientation,
    action rate, etc.) are active. The robot first learns to stand upright
    without falling.

Phase 2 — Targeting
    Once the per-iteration stability reward plateaus (plateau detector
    below), ``target_coeff`` ramps from ``0`` to ``max_target_coeff``, so
    ``tracking_target`` + ``tracking_yaw`` start dominating. The robot then
    learns to walk to the red virtual ball and face the commanded yaw.

Usage::

    uv run src/examples/b2_train.py
    uv run src/examples/b2_train.py -e b2-custom --max_iterations 800 -B 4096
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import time
from collections import deque
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
from rsl_rl.utils import store_code_state

import genesis as gs

from b2_env import B2TargetEnv, default_cfgs


# =========================================================================
# Plateau detector (copied verbatim from dynamic_ics_dog_train.py)
# =========================================================================
def detect_plateau(reward_history, window: int = 50, threshold: float = 0.05):
    if len(reward_history) < window:
        return False, {}
    recent = torch.tensor(list(reward_history)[-window:])
    gradients = recent[1:] - recent[:-1]
    mean_reward = recent.mean().item()
    gradient_std = gradients.std().item()
    normalized_gradient_std = gradient_std / max(abs(mean_reward), 1e-6)
    return normalized_gradient_std < threshold, {
        "mean_reward": mean_reward,
        "gradient_std": gradient_std,
        "normalized_gradient_std": normalized_gradient_std,
    }


# =========================================================================
# Training loop with dynamic coefficient scheduling
# =========================================================================
def dynamic_learn(runner, env, max_iterations, dynamic_cfg, init_at_random_ep_len=True):
    plateau_window = dynamic_cfg.get("plateau_window", 50)
    plateau_threshold = dynamic_cfg.get("plateau_threshold", 0.05)
    plateau_patience = dynamic_cfg.get("plateau_patience", 20)
    coeff_increment = dynamic_cfg.get("coeff_increment", 0.1)
    max_target_coeff = dynamic_cfg.get("max_target_coeff", 1.0)
    check_interval = dynamic_cfg.get("check_interval", 5)
    min_iterations = dynamic_cfg.get("min_iterations", 50)

    if runner.log_dir is not None and runner.writer is None:
        runner.logger_type = runner.cfg.get("logger", "tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        runner.writer = SummaryWriter(log_dir=runner.log_dir, flush_secs=10)

    if init_at_random_ep_len:
        env.episode_length_buf = torch.randint_like(
            env.episode_length_buf, high=int(env.max_episode_length)
        )

    obs, extras = env.get_observations()
    critic_obs = extras["observations"].get("critic", obs)
    obs, critic_obs = obs.to(runner.device), critic_obs.to(runner.device)
    runner.train_mode()

    ep_infos = []
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=runner.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=runner.device)

    if runner.alg.rnd:
        erewbuffer = deque(maxlen=100)
        irewbuffer = deque(maxlen=100)
        cur_ereward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=runner.device)
        cur_ireward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=runner.device)

    stab_reward_history = deque(maxlen=plateau_window)
    plateau_count = 0
    phase = "stability"

    start_iter = runner.current_learning_iteration
    tot_iter = start_iter + max_iterations
    num_learning_iterations = max_iterations

    for it in range(start_iter, tot_iter):
        start = time.time()
        with torch.inference_mode():
            for _ in range(runner.num_steps_per_env):
                actions = runner.alg.act(obs, critic_obs)
                obs, rewards, dones, infos = env.step(actions.to(env.device))
                obs = obs.to(runner.device)
                rewards = rewards.to(runner.device)
                dones = dones.to(runner.device)
                obs = runner.obs_normalizer(obs)
                if "critic" in infos["observations"]:
                    critic_obs = runner.critic_obs_normalizer(
                        infos["observations"]["critic"].to(runner.device)
                    )
                else:
                    critic_obs = obs
                runner.alg.process_env_step(rewards, dones, infos)
                intrinsic_rewards = runner.alg.intrinsic_rewards if runner.alg.rnd else None
                if runner.log_dir is not None:
                    if "episode" in infos:
                        ep_infos.append(infos["episode"])
                    elif "log" in infos:
                        ep_infos.append(infos["log"])
                    if runner.alg.rnd:
                        cur_ereward_sum += rewards
                        cur_ireward_sum += intrinsic_rewards
                        cur_reward_sum += rewards + intrinsic_rewards
                    else:
                        cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0
                    if runner.alg.rnd:
                        erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        cur_ereward_sum[new_ids] = 0
                        cur_ireward_sum[new_ids] = 0
            stop = time.time()
            collection_time = stop - start

            start = stop
            runner.alg.compute_returns(critic_obs)

        mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss = (
            runner.alg.update()
        )
        stop = time.time()
        learn_time = stop - start
        runner.current_learning_iteration = it

        if runner.log_dir is not None:
            runner.log(locals())
            runner.writer.add_scalar("Dynamic/target_coeff", env.target_coeff, it)
            runner.writer.add_scalar("Dynamic/phase", 0 if phase == "stability" else 1, it)
            if it % runner.save_interval == 0:
                runner.save(os.path.join(runner.log_dir, f"model_{it}.pt"))
                pickle.dump(
                    {"target_coeff": env.target_coeff, "phase": phase, "iteration": it},
                    open(os.path.join(runner.log_dir, f"dynamic_state_{it}.pkl"), "wb"),
                )

        if ep_infos:
            stab_reward_accum = 0.0
            for ep_info in ep_infos:
                for key, val in ep_info.items():
                    if not key.startswith("rew_"):
                        continue
                    if "tracking_target" in key or "tracking_yaw" in key:
                        continue
                    stab_reward_accum += val.item() if isinstance(val, torch.Tensor) else float(val)
            stab_reward_history.append(stab_reward_accum / len(ep_infos))

        if it >= min_iterations and it % check_interval == 0:
            if phase == "stability":
                is_plateau, stats = detect_plateau(
                    stab_reward_history,
                    window=plateau_window,
                    threshold=plateau_threshold,
                )
                plateau_count = (plateau_count + 1) if is_plateau else max(0, plateau_count - 1)
                if runner.log_dir is not None and stats:
                    runner.writer.add_scalar("Dynamic/gradient_std", stats.get("gradient_std", 0), it)
                    runner.writer.add_scalar(
                        "Dynamic/norm_gradient_std",
                        stats.get("normalized_gradient_std", 0),
                        it,
                    )
                    runner.writer.add_scalar("Dynamic/plateau_count", plateau_count, it)
                if plateau_count >= plateau_patience:
                    phase = "targeting"
                    print(
                        f"\n{'=' * 60}\n"
                        f" PHASE TRANSITION: stability -> targeting  (iter {it})\n"
                        f"{'=' * 60}\n"
                    )
            if phase == "targeting" and env.target_coeff < max_target_coeff:
                env.target_coeff = min(env.target_coeff + coeff_increment, max_target_coeff)
                print(f"  [targeting] target_coeff = {env.target_coeff:.3f}")

        if it == start_iter:
            try:
                git_file_paths = store_code_state(
                    runner.log_dir, runner.git_status_repos
                )
            except Exception as exc:  # bare repo, no commits, etc.
                print(f"[b2_train] skipping git snapshot: {exc}")
                git_file_paths = []
            if runner.logger_type in ("wandb", "neptune") and git_file_paths:
                for p in git_file_paths:
                    runner.writer.save_file(p)
        ep_infos.clear()

    if runner.log_dir is not None:
        final_it = runner.current_learning_iteration
        runner.save(os.path.join(runner.log_dir, f"model_{final_it}.pt"))
        pickle.dump(
            {"target_coeff": env.target_coeff, "phase": phase, "iteration": final_it},
            open(os.path.join(runner.log_dir, f"dynamic_state_{final_it}.pkl"), "wb"),
        )


# =========================================================================
# Runner configuration
# =========================================================================
def get_train_cfg(exp_name: str, max_iterations: int) -> dict:
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


# =========================================================================
# Main
# =========================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="b2-target-rl")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument(
        "--backend",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
        help="Genesis backend. Use 'cpu' for smoke-tests on machines without CUDA.",
    )
    args = parser.parse_args()

    env_cfg, obs_cfg, reward_cfg, command_cfg, target_cfg, dynamic_cfg = default_cfgs()
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
    gs.init(
        backend=backend,
        precision="32",
        logging_level="warning",
        seed=train_cfg["seed"],
        performance_mode=True,
    )

    env = B2TargetEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        target_cfg=target_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    dynamic_learn(runner, env, args.max_iterations, dynamic_cfg, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
