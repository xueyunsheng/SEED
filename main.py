"""SEED online: SAC with Koopman spectral features and adaptive categorical value learning.

The offline transition dataset is used only to pretrain the frozen Koopman encoder.
"""
import os
import random
import time
from dataclasses import dataclass
from typing import Optional
from functools import partial

import gymnasium as gym
import numpy as np
import tyro
from torch.utils.tensorboard import SummaryWriter

import jax
import jax.numpy as jnp
from flax.training import train_state
from flax.core import FrozenDict
import optax

from agents.seed import Actor, DoubleCritic, Temperature, TargetTrainState, sample_action, train_step
from utils.koopman import KoopmanBilinear, KoopmanEncoder, koopman_train_step, solve_eig_numpy
from utils.distribution import HLGaussWithXi, get_expectation_from_probs
from utils.replay_buffer import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    track: bool = False
    wandb_project_name: str = "Online_RL"
    wandb_entity: Optional[str] = None
    wandb_group: str = "SEED_Online"
    capture_video: bool = False

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    total_timesteps: int = 1000000
    num_envs: int = 1
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5000
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1 
    alpha: float = 0.2
    autotune: bool = True

    num_blocks: int = 2           
    hidden_dim: int = 256         

    num_bins: int = 51       
    sigma: float = 1.5        
    xi_init: float = 10.0    
    ahl_lr: float = 1e-2      

    sigma_times: float = 4.0     
    lambda_weight: float = 1.0   
    
    offline_dataset_steps: int = 100000
    dataset_dir: str = "datasets"
    koopman_latent_dim: int = 32
    koopman_hidden_dim: int = 256
    k_eig_dim: int = 24
    koopman_lr: float = 3e-4
    koopman_train_steps: int = 30000 


class NormalizeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, mean, std):
        super().__init__(env)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def observation(self, observation):
        return (observation - self.mean) / self.std

def make_env(env_id, seed, idx, capture_video, run_name, state_mean, state_std):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NormalizeObservationWrapper(env, state_mean, state_std)
        env.action_space.seed(seed)
        return env
    return thunk

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.num_envs != 1:
        raise ValueError("This replay buffer currently supports --num-envs 1 only")
    if args.num_bins < 2 or args.sigma <= 0 or args.xi_init <= 0:
        raise ValueError("num_bins >= 2, sigma > 0 and xi_init > 0 are required")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name, entity=args.wandb_entity,
            group=args.wandb_group, sync_tensorboard=True, config=vars(args), 
            name=run_name, monitor_gym=True, save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    
    dataset_path = os.path.join(args.dataset_dir, f"{args.env_id}_step_{args.offline_dataset_steps}.npz")
    print(f"Loading offline dataset for Koopman from: {dataset_path}")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Koopman pretraining dataset missing: {dataset_path}. See README.md")
    
    offline_data = np.load(dataset_path)
    offline_obs, offline_actions, offline_next_obs = offline_data['observations'], offline_data['actions'], offline_data['next_observations']
    if not (len(offline_obs) == len(offline_actions) == len(offline_next_obs)) or len(offline_obs) == 0:
        raise ValueError("Dataset observations, actions and next_observations must have equal, nonzero length")
    state_dim, action_dim = offline_obs.shape[-1], offline_actions.shape[-1]
    
    state_mean = offline_obs.mean(axis=0)
    state_std = offline_obs.std(axis=0) + 1e-3
    norm_obs = (offline_obs - state_mean) / state_std
    norm_next_obs = (offline_next_obs - state_mean) / state_std
    dataset_size = norm_obs.shape[0]

    rng, koop_key = jax.random.split(rng)
    dummy_state = jnp.ones((1, state_dim))
    dummy_action = jnp.ones((1, action_dim))

    print("--- Starting Koopman Pre-training ---")
    koopman_def = KoopmanBilinear(
        latent_dim=args.koopman_latent_dim, action_dim=action_dim, 
        state_dim=state_dim, hidden_dim=args.koopman_hidden_dim, obs_lower=-10.0, obs_upper=10.0 
    )
    koopman_params = koopman_def.init(koop_key, dummy_state, dummy_action)['params']
    koopman_state = train_state.TrainState.create(apply_fn=koopman_def.apply, params=koopman_params, tx=optax.adam(learning_rate=args.koopman_lr))

    for step in range(args.koopman_train_steps):
        indices = np.random.randint(0, dataset_size, size=args.batch_size)
        koopman_state, k_logs = koopman_train_step(norm_obs[indices], offline_actions[indices], norm_next_obs[indices], koopman_state, koopman_def)
        if step % 100 == 0:
            writer.add_scalar("koopman_pretrain/total_loss", k_logs['koopman_total_loss'].item(), step)
            writer.add_scalar("koopman_pretrain/pred_loss", k_logs['koopman_pred_loss'].item(), step)
            writer.add_scalar("koopman_pretrain/recon_loss", k_logs['koopman_recon_loss'].item(), step)
        if (step + 1) % 5000 == 0:
            print(f"Koopman Step {step+1}/{args.koopman_train_steps}, Loss: {k_logs['koopman_total_loss']:.4f}")
            writer.flush()

    layerK_weights = jax.device_get(koopman_state.params['layerK']['kernel']).T
    eigenvectors = jnp.array(solve_eig_numpy(layerK_weights, args.k_eig_dim))
    koop_encoder_params = FrozenDict(koopman_state.params['encoder'])
    encoder_def = KoopmanEncoder(args.koopman_latent_dim, args.koopman_hidden_dim)
    print("--- Koopman Pre-training Finished! ---")


    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, state_mean, state_std) for i in range(args.num_envs)])
    
    rng, actor_key, critic_key, alpha_key, ahl_key = jax.random.split(rng, 5)

    action_scale = jnp.array((envs.single_action_space.high - envs.single_action_space.low) / 2.0)
    action_bias = jnp.array((envs.single_action_space.high + envs.single_action_space.low) / 2.0)
    dummy_zs = jnp.ones((1, args.k_eig_dim * 2))

    actor_def = Actor(action_dim=action_dim)
    actor_params = actor_def.init(actor_key, dummy_state, dummy_zs)['params']
    actor_state = train_state.TrainState.create(apply_fn=actor_def.apply, params=actor_params, tx=optax.adam(learning_rate=args.policy_lr))

    critic_def = DoubleCritic(hidden_dim=args.hidden_dim, hidden_blocks=args.num_blocks, num_bins=args.num_bins)
    critic_params = critic_def.init(critic_key, dummy_state, dummy_action, dummy_zs)['params']
    critic_state = TargetTrainState.create(apply_fn=critic_def.apply, params=critic_params, target_params=critic_params, tx=optax.adam(learning_rate=args.q_lr))

    alpha_def = Temperature(initial_value=args.alpha)
    alpha_params = alpha_def.init(alpha_key)['params']
    alpha_state = train_state.TrainState.create(apply_fn=alpha_def.apply, params=alpha_params, tx=optax.adam(learning_rate=args.q_lr))

    ahl_def = HLGaussWithXi(num_bins=args.num_bins, sigma_scale=args.sigma, xi_init=args.xi_init)
    ahl_params = ahl_def.init(ahl_key, jnp.zeros((1, 1)))['params']
    ahl_state = train_state.TrainState.create(apply_fn=ahl_def.apply, params=ahl_params, tx=optax.adam(learning_rate=args.ahl_lr))

    target_entropy = -jnp.prod(jnp.array(envs.single_action_space.shape)).item()
    static_config = FrozenDict({
            'gamma': args.gamma, 'tau': args.tau, 'alpha': args.alpha, 'num_bins': args.num_bins,
            'policy_frequency': args.policy_frequency, 'target_network_frequency': args.target_network_frequency,
            'autotune': args.autotune, 'target_entropy': target_entropy,
            'sigma_times': args.sigma_times, 'lambda_weight': args.lambda_weight
        })
    rb = ReplayBuffer(state_dim, action_dim, args.buffer_size)
    start_time = time.time()

    @jax.jit
    def get_eval_action(rng, a_params, state, zs, action_scale, action_bias):
        action, _ = sample_action(rng, a_params, actor_def, state, zs, action_scale, action_bias)
        return action

    @partial(jax.jit, static_argnames=('num_bins',))
    def get_q_value(critic_params, ahl_params, state, action, zs, num_bins):
        _, _, current_xi = ahl_def.apply({'params': ahl_params}, jnp.zeros((1, 1)))
        
        q1_logits, q2_logits = critic_def.apply({'params': critic_params}, state, action, zs)
        q1_probs = jax.nn.softmax(q1_logits, axis=-1)
        q2_probs = jax.nn.softmax(q2_logits, axis=-1)
        
        q1_val = get_expectation_from_probs(q1_probs, current_xi, num_bins)
        q2_val = get_expectation_from_probs(q2_probs, current_xi, num_bins)
        
        return jnp.minimum(q1_val, q2_val)

    episode_start_q_preds = np.full(args.num_envs, np.nan, dtype=np.float32)
    episode_discounted_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_step_counts = np.zeros(args.num_envs, dtype=np.int32)
    is_start_of_episode = np.ones(args.num_envs, dtype=bool)

    obs, _ = envs.reset(seed=args.seed)
    
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            rng, eval_key = jax.random.split(rng)
            batched_obs = jnp.array(obs).reshape(args.num_envs, -1)
            latent_z_eval = encoder_def.apply({'params': koop_encoder_params}, batched_obs)
            zs_eval = latent_z_eval @ eigenvectors
            actions_jax = get_eval_action(eval_key, actor_state.params, batched_obs, zs_eval, action_scale, action_bias)
            actions = np.asarray(actions_jax)

        for i in range(args.num_envs):
            if is_start_of_episode[i]:
                if global_step >= args.learning_starts:
                    single_obs = batched_obs[i:i+1]
                    single_action = actions_jax[i:i+1]
                    single_zs = zs_eval[i:i+1]
                    q_val = get_q_value(critic_state.params, ahl_state.params, single_obs, single_action, single_zs, args.num_bins)
                    episode_start_q_preds[i] = np.asarray(q_val)[0, 0]
                is_start_of_episode[i] = False

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        for i in range(args.num_envs):
            episode_discounted_returns[i] += rewards[i] * (args.gamma ** episode_step_counts[i])
            episode_step_counts[i] += 1

            if terminations[i] or truncations[i]:
                if not np.isnan(episode_start_q_preds[i]):
                    true_mc_return = episode_discounted_returns[i]
                    q_prediction = episode_start_q_preds[i]
                    
                    estimation_bias = q_prediction - true_mc_return 
                    
                    writer.add_scalar("evaluation/True_MC_Return_G0", true_mc_return, global_step)
                    writer.add_scalar("evaluation/Predicted_Q0", q_prediction, global_step)
                    writer.add_scalar("evaluation/Estimation_Bias(Q-MC)", estimation_bias, global_step)
                
                episode_discounted_returns[i] = 0.0
                episode_step_counts[i] = 0
                episode_start_q_preds[i] = np.nan
                is_start_of_episode[i] = True

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                if "final_observation" in infos and infos["final_observation"] is not None:
                    if infos["final_observation"][idx] is not None:
                        real_next_obs[idx] = infos["final_observation"][idx]
                elif "final_info" in infos and infos["final_info"] is not None:
                    if infos["final_info"][idx] is not None and "observation" in infos["final_info"][idx]:
                        real_next_obs[idx] = infos["final_info"][idx]["observation"]
                
        rb.add(obs[0], real_next_obs[0], actions[0], rewards[0], terminations[0], infos)
        obs = next_obs

        if global_step > args.learning_starts:
            batch = rb.sample(args.batch_size)
            
            rng, actor_state, critic_state, alpha_state, ahl_state, jax_log_dict = train_step(
                rng, actor_state, critic_state, alpha_state, ahl_state,
                koop_encoder_params, eigenvectors, batch, global_step, 
                action_scale, action_bias, actor_def, encoder_def, ahl_def, static_config
            )

            if global_step % 100 == 0:
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("charts/SPS", sps, global_step)
                for k, v in jax_log_dict.items():
                    writer.add_scalar(k, v.item(), global_step)

    envs.close()
    writer.close()
    if args.track:
        wandb.finish()
