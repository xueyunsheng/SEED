"""Bilinear Koopman dynamics and frozen slow-mode extraction."""
from functools import partial
import numpy as np
import scipy.linalg as linalg
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state

class KoopmanEncoder(nn.Module):
    latent_dim: int
    hidden_dim: int = 256
    @nn.compact
    def __call__(self, state):
        x = nn.tanh(nn.Dense(self.hidden_dim)(state))
        x = nn.tanh(nn.Dense(self.hidden_dim)(x))
        return nn.tanh(nn.Dense(self.latent_dim)(x))

class KoopmanDecoder(nn.Module):
    state_dim: int
    hidden_dim: int = 256
    obs_lower: float = -1.0
    obs_upper: float = 1.0
    @nn.compact
    def __call__(self, g_state):
        x = nn.tanh(nn.Dense(self.hidden_dim)(g_state))
        x = nn.tanh(nn.Dense(self.hidden_dim)(x))
        x = nn.Dense(self.state_dim)(x)
        return jnp.clip(x, self.obs_lower, self.obs_upper)

class KoopmanBilinear(nn.Module):
    latent_dim: int
    action_dim: int
    state_dim: int
    hidden_dim: int = 256
    obs_lower: float = -1.0
    obs_upper: float = 1.0
    def setup(self):
        self.encoder = KoopmanEncoder(self.latent_dim, self.hidden_dim)
        self.decoder = KoopmanDecoder(self.state_dim, self.hidden_dim, self.obs_lower, self.obs_upper)
        self.layerK = nn.Dense(self.latent_dim, use_bias=False, name='layerK')
        self.layerBi = nn.Dense(self.latent_dim * self.action_dim, use_bias=False, name='layerBi')
    def __call__(self, state, action):
        gt = self.encoder(state)
        gt_K = self.layerK(gt)
        gt_Bi = self.layerBi(gt).reshape(-1, self.action_dim, self.latent_dim)
        control_term = jnp.sum(action[..., None] * gt_Bi, axis=1)
        gtp1 = gt_K + control_term
        return self.decoder(gtp1), self.decoder(gt)

@partial(jax.jit, static_argnames=('model',))
def koopman_train_step(state, action, next_state, train_state, model):
    def loss_fn(params):
        pred_next, recon = model.apply({'params': params}, state, action)
        pred_loss = jnp.mean((pred_next - next_state)**2)
        recon_loss = jnp.mean((recon - state)**2)
        total_loss = pred_loss + 0.1 * recon_loss
        return total_loss, (pred_loss, recon_loss)
    (loss, (pred_loss, recon_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, {"koopman_total_loss": loss, "koopman_pred_loss": pred_loss, "koopman_recon_loss": recon_loss}

def solve_eig_numpy(K_matrix: np.ndarray, k_eig_dim: int) -> np.ndarray:
    dim = K_matrix.shape[0]
    try:
        eigenvalues, U = linalg.eig(K_matrix)
    except linalg.LinAlgError as exc:
        raise ValueError("Koopman drift eigendecomposition failed") from exc
    if not 0 < k_eig_dim <= dim:
        raise ValueError(f"k_eig_dim must be in [1, {dim}]")
    distances_from_1 = np.abs(np.abs(eigenvalues) - 1)
    sorted_indices = np.argsort(distances_from_1)
    sorted_U = U[:, sorted_indices]
    top_k_complex = sorted_U[:, :k_eig_dim]
    top_k_eigenvectors = np.concatenate([np.real(top_k_complex), np.imag(top_k_complex)], axis=1)
    return top_k_eigenvectors

