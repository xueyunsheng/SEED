"""SEED SAC actor, categorical critic, and update step."""
from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from flax.core import FrozenDict
from utils.distribution import get_expectation_from_probs

class ResidualBlock(nn.Module):
    out_features: int
    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]
        shortcut = x if in_features == self.out_features else nn.Dense(self.out_features)(x)
        out = nn.silu(nn.LayerNorm()(nn.Dense(self.out_features)(x)))
        out = nn.Dense(self.out_features)(out)
        return out + shortcut

class Critic(nn.Module):
    hidden_dim: int = 512
    hidden_blocks: int = 5
    num_bins: int = 51 

    @nn.compact
    def __call__(self, state, action, zs):
        sa = jnp.concatenate([state, action], axis=-1)
        x = nn.silu(nn.Dense(self.hidden_dim)(sa))
        for _ in range(self.hidden_blocks):
            x = ResidualBlock(self.hidden_dim)(x)
            
        z_feat = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal())(zs)
        z_feat = nn.LayerNorm()(z_feat)
        z_feat = nn.silu(z_feat)
        
        combined = jnp.concatenate([x, z_feat], axis=-1)
        return nn.Dense(self.num_bins)(combined)

class DoubleCritic(nn.Module):
    hidden_dim: int = 512
    hidden_blocks: int = 5
    num_bins: int = 51

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray, zs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q1 = Critic(self.hidden_dim, self.hidden_blocks, self.num_bins, name='critic_1')(state, action, zs)
        q2 = Critic(self.hidden_dim, self.hidden_blocks, self.num_bins, name='critic_2')(state, action, zs)
        return q1, q2

class Actor(nn.Module):
    action_dim: int
    LOG_STD_MAX: float = 2.0
    LOG_STD_MIN: float = -5.0
    @nn.compact
    def __call__(self, state, zs):
        s_emb = nn.relu(nn.Dense(256)(state))
        z_emb = nn.Dense(256, kernel_init=nn.initializers.orthogonal())(zs)
        z_emb = nn.LayerNorm()(z_emb)
        z_emb = nn.silu(z_emb)
        combined = jnp.concatenate([s_emb, z_emb], axis=-1)
        x = nn.relu(nn.Dense(256)(combined))
        x = nn.relu(nn.Dense(256)(x))
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

class Temperature(nn.Module):
    initial_value: float = 1.0
    @nn.compact
    def __call__(self):
        log_alpha = self.param('log_alpha', init_fn=lambda key: jnp.full((), jnp.log(self.initial_value)))
        return jnp.exp(log_alpha), log_alpha

class TargetTrainState(train_state.TrainState):
    target_params: FrozenDict = None

@partial(jax.jit, static_argnames=('actor_def',))
def sample_action(rng, actor_params, actor_def, state, zs, action_scale, action_bias):
    mean, log_std = actor_def.apply({'params': actor_params}, state, zs)
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng, mean.shape)
    x_t = mean + noise * std
    y_t = jnp.tanh(x_t)
    action = y_t * action_scale + action_bias
    log_prob = jax.scipy.stats.norm.logpdf(x_t, mean, std)
    log_prob -= jnp.log(action_scale * (1 - y_t**2) + 1e-6)
    log_prob = log_prob.sum(axis=-1, keepdims=True)
    return action, log_prob

def soft_update(target_params, online_params, tau):
    return jax.tree_util.tree_map(lambda t, o: (1 - tau) * t + tau * o, target_params, online_params)

@partial(jax.jit, static_argnames=('actor_def', 'config', 'encoder_def', 'ahl_def'))
def train_step(
    rng: jnp.ndarray,
    actor_state: train_state.TrainState,
    critic_state: TargetTrainState,
    alpha_state: train_state.TrainState,
    ahl_state: train_state.TrainState,
    koop_encoder_params: FrozenDict,
    eigenvectors: jnp.ndarray,
    batch: Tuple[jnp.ndarray, ...],
    total_it: int,
    action_scale: jnp.ndarray,
    action_bias: jnp.ndarray,
    actor_def: nn.Module,
    encoder_def: nn.Module,
    ahl_def: nn.Module,
    config: FrozenDict
):
    state, action, reward, next_state, done = batch
    reward = reward.reshape(-1, 1)
    not_done = (1.0 - done).reshape(-1, 1)
    rng, action_key = jax.random.split(rng)

    alpha, _ = alpha_state.apply_fn({'params': alpha_state.params})
    alpha = jax.lax.stop_gradient(alpha) if config['autotune'] else config['alpha']

    latent_z = encoder_def.apply({'params': koop_encoder_params}, state)
    zs = latent_z @ eigenvectors
    latent_next_z = encoder_def.apply({'params': koop_encoder_params}, next_state)
    next_zs = latent_next_z @ eigenvectors

    _, _, current_xi = ahl_def.apply({'params': ahl_state.params}, jnp.zeros((1, 1)))

    next_action, next_log_prob = sample_action(action_key, actor_state.params, actor_def, next_state, next_zs, action_scale, action_bias)
    
    target_q1_logits, target_q2_logits = critic_state.apply_fn({'params': critic_state.target_params}, next_state, next_action, next_zs)
    target_q1_probs = jax.nn.softmax(target_q1_logits, axis=-1)
    target_q2_probs = jax.nn.softmax(target_q2_logits, axis=-1)
    
    target_q1_val = get_expectation_from_probs(target_q1_probs, current_xi, config['num_bins'])
    target_q2_val = get_expectation_from_probs(target_q2_probs, current_xi, config['num_bins'])
    
    target_q_scalar = reward + not_done * config['gamma'] * (jnp.minimum(target_q1_val, target_q2_val) - alpha * next_log_prob)
    
    truncation_error = jnp.mean((jnp.abs(target_q_scalar) > current_xi).astype(jnp.float32))

    def ahl_loss_fn(ahl_params):
            target_probs, predict, xi = ahl_def.apply({'params': ahl_params}, target_q_scalar)
            
            mse_loss = jnp.mean((predict - target_q_scalar)**2)
            
            mu = jnp.mean(target_q_scalar)
            sigma = jnp.std(target_q_scalar)
            confidence_loss = jax.nn.relu((jnp.abs(mu) + config['sigma_times'] * sigma) - xi[0])
            
            total_boundary_loss = mse_loss + config['lambda_weight'] * confidence_loss
            
            return total_boundary_loss, (target_probs, xi, mse_loss, confidence_loss, mu, sigma)
            
    (ahl_total_loss, (td_target_probs, updated_xi, ahl_mse_loss, ahl_conf_loss, q_mu, q_sigma)), ahl_grads = jax.value_and_grad(ahl_loss_fn, has_aux=True)(ahl_state.params)
    ahl_state = ahl_state.apply_gradients(grads=ahl_grads)
    td_target_probs, _, updated_xi = ahl_def.apply({'params': ahl_state.params}, target_q_scalar)

    def critic_loss_fn(c_params):
        q1_logits, q2_logits = critic_state.apply_fn({'params': c_params}, state, action, zs)
        q1_log_probs = jax.nn.log_softmax(q1_logits, axis=-1)
        q2_log_probs = jax.nn.log_softmax(q2_logits, axis=-1)
        stopped_target_probs = jax.lax.stop_gradient(td_target_probs)
        

        loss1 = -jnp.sum(stopped_target_probs * q1_log_probs, axis=-1).mean()
        loss2 = -jnp.sum(stopped_target_probs * q2_log_probs, axis=-1).mean()
        return loss1 + loss2, (q1_logits, q2_logits)

    (critic_loss, (q1_logits, q2_logits)), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_state.params)
    critic_state = critic_state.apply_gradients(grads=critic_grads)

    def update_actor_alpha(state_tuple):
        act_st, alp_st, crit_st = state_tuple
        rng_inner = jax.random.fold_in(action_key, total_it) 
        final_actor_loss, final_alpha_loss, final_alpha = jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)

        for _ in range(1):
            rng_inner, actor_key, alpha_key = jax.random.split(rng_inner, 3)
            
            def actor_loss_fn(a_params):
                pi, log_pi = sample_action(actor_key, a_params, actor_def, state, zs, action_scale, action_bias)
                q1_logits_pi, q2_logits_pi = crit_st.apply_fn({'params': crit_st.params}, state, pi, zs)
                
                q1_probs_pi = jax.nn.softmax(q1_logits_pi, axis=-1)
                q2_probs_pi = jax.nn.softmax(q2_logits_pi, axis=-1)
                q1_val_pi = get_expectation_from_probs(q1_probs_pi, updated_xi, config['num_bins'])
                q2_val_pi = get_expectation_from_probs(q2_probs_pi, updated_xi, config['num_bins'])
                
                min_q_pi = jnp.minimum(q1_val_pi, q2_val_pi)
                curr_alpha = jnp.exp(alp_st.apply_fn({'params': alp_st.params})[1]) if config['autotune'] else config['alpha']
                loss = jnp.mean(jax.lax.stop_gradient(curr_alpha) * log_pi - min_q_pi)
                return loss

            actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(act_st.params)
            act_st = act_st.apply_gradients(grads=actor_grads)
            final_actor_loss = actor_loss

            if config['autotune']:
                _, new_log_pi = sample_action(alpha_key, act_st.params, actor_def, state, zs, action_scale, action_bias)
                def alpha_loss_fn(alp_params):
                    curr_alpha, _ = alp_st.apply_fn({'params': alp_params})
                    loss = jnp.mean(-curr_alpha * jax.lax.stop_gradient(new_log_pi + config['target_entropy']))
                    return loss, curr_alpha
                (alpha_loss, current_alpha), alpha_grads = jax.value_and_grad(alpha_loss_fn, has_aux=True)(alp_st.params)
                alp_st = alp_st.apply_gradients(grads=alpha_grads)
                final_alpha_loss = alpha_loss
                final_alpha = current_alpha
            else:
                final_alpha = jnp.array(config['alpha'])

        return act_st, alp_st, final_actor_loss, final_alpha_loss, final_alpha

    def skip_actor_alpha(state_tuple):
        act_st, alp_st, crit_st = state_tuple
        return act_st, alp_st, jnp.array(0.0), jnp.array(0.0), alpha

    actor_state, alpha_state, actor_loss, alpha_loss, current_alpha = jax.lax.cond(
        total_it % config['policy_frequency'] == 0,
        update_actor_alpha, skip_actor_alpha, (actor_state, alpha_state, critic_state)
    )

    critic_state = jax.lax.cond(
        total_it % config['target_network_frequency'] == 0,
        lambda cs: cs.replace(target_params=soft_update(cs.target_params, cs.params, config['tau'])),
        lambda cs: cs,
        critic_state
    )

    log_dict = {
            "losses/qf_loss(CrossEntropy)": critic_loss / 2.0,
            "losses/AHL_total_loss": ahl_total_loss,
            "losses/AHL_mse_loss": ahl_mse_loss,               
            "losses/AHL_confidence_loss": ahl_conf_loss,       
            "losses/Boundary_Value(xi)": updated_xi[0],
            "losses/target_q_mu": q_mu,                         
            "losses/target_q_sigma": q_sigma,                  
            "losses/truncation_error": truncation_error,
            "losses/actor_loss": actor_loss,
            "losses/alpha_loss": alpha_loss,
            "losses/alpha": current_alpha
        }

    return rng, actor_state, critic_state, alpha_state, ahl_state, log_dict

