"""Adaptive categorical support with a truncated Gaussian target."""
import jax
import jax.numpy as jnp
import flax.linen as nn

class HLGaussWithXi(nn.Module):
    num_bins: int
    sigma_scale: float
    xi_init: float

    @nn.compact
    def __call__(self, target_q: jnp.ndarray):
        raw_xi = self.param('raw_xi', lambda rng: jnp.array([self.xi_init], dtype=jnp.float32))
        xi = nn.softplus(raw_xi)
        
        w_xi = 2 * xi / self.num_bins
        idx = jnp.arange(self.num_bins, dtype=jnp.float32).reshape(1, -1)
        centers = -xi + w_xi / 2 + idx * w_xi
        
        sigma_xi = self.sigma_scale * w_xi
        sqrt_2 = jnp.sqrt(2.0)
        
        x = target_q.reshape(-1, 1)
        
        arg1 = (centers + w_xi / 2 - x) / (sqrt_2 * sigma_xi)
        arg2 = (centers - w_xi / 2 - x) / (sqrt_2 * sigma_xi)
        p = 0.5 * (jax.scipy.special.erf(arg1) - jax.scipy.special.erf(arg2))
        
        arg3 = (xi - x) / (sqrt_2 * sigma_xi)
        arg4 = (-xi - x) / (sqrt_2 * sigma_xi)
        Z_xi = 0.5 * (jax.scipy.special.erf(arg3) - jax.scipy.special.erf(arg4))
        
        p = p / (Z_xi + 1e-8)
        expectation = jnp.sum(p * centers, axis=1, keepdims=True)
        return p, expectation, xi

def get_expectation_from_probs(probs, xi, num_bins):
    w_xi = 2 * xi / num_bins
    idx = jnp.arange(num_bins, dtype=jnp.float32).reshape(1, -1)
    centers = -xi + w_xi / 2 + idx * w_xi
    return jnp.sum(probs * centers, axis=-1, keepdims=True)

