from typing import Any, Callable, Mapping, Sequence, Union

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen
import jax
import jax.numpy as jnp


_PROP_SLICE = slice(0, 52)
_ACTION_SLICE = slice(52, 76)
_TASK_SLICES = (
    slice(76, 85),    # command, foot height, gait phase
    slice(85, 92),    # head
    slice(92, 99),    # pelvis
    slice(99, 106),   # torso
    slice(106, 120),  # feet
    slice(120, 134),  # hands
    slice(134, 148),  # knees
    slice(148, 162),  # shoulders
)


def _get_obs_state_size(obs_size: types.ObservationSize, obs_key: str) -> int:
    if isinstance(obs_size, Mapping):
        obs_size = obs_size[obs_key]
    if isinstance(obs_size, Sequence):
        if len(obs_size) != 1:
            raise ValueError(f"Expected flat observation for {obs_key}, got {obs_size}")
        return int(obs_size[0])
    return int(obs_size)


class RMSNorm(linen.Module):
    dim: int
    eps: float = 1e-8

    @linen.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        scale = self.param("scale", linen.initializers.ones, (self.dim,))
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True) / jnp.sqrt(self.dim)
        return scale * x / (norm + self.eps)


class SwiGLU(linen.Module):
    hidden_dim: int

    @linen.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        gate = linen.silu(linen.Dense(self.hidden_dim, use_bias=False, name="w")(x))
        value = linen.Dense(self.hidden_dim, use_bias=False, name="v")(x)
        return linen.Dense(x.shape[-1], use_bias=False, name="output")(gate * value)


class HumanoidTransformerBlock(linen.Module):
    embed_dim: int
    num_heads: int
    ff_dim: int

    @linen.compact
    def __call__(self, x: jax.Array, task_tokens: jax.Array) -> jax.Array:
        self_mask = jnp.array(
            [
                [True, True, False],
                [True, True, False],
                [True, True, True],
            ],
            dtype=bool,
        )[None, None, :, :]

        x_norm = RMSNorm(self.embed_dim, name="rmsnorm1")(x)
        attn_output = linen.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            name="self_attention",
        )(x_norm, x_norm, mask=self_mask, deterministic=True)
        x = x + attn_output

        x_norm = RMSNorm(self.embed_dim, name="rmsnorm2")(x)
        task_norm = RMSNorm(self.embed_dim, name="cond_norm")(task_tokens)
        cross_output = linen.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            name="cross_attention",
        )(x_norm, task_norm, deterministic=True)
        x = x + cross_output

        x_norm = RMSNorm(self.embed_dim, name="rmsnorm3")(x)
        x = x + SwiGLU(self.ff_dim, name="feed_forward")(x_norm)
        return x


class HumanoidTransformerPolicy(linen.Module):
    output_dim: int
    embed_dim: int = 256
    num_heads: int = 4
    ff_dim: int = 512
    num_layers: int = 4

    @linen.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        if obs.shape[-1] != 162:
            raise ValueError(f"HumanoidTransformerPolicy expects state dim 162, got {obs.shape[-1]}")

        prop_obs = obs[..., _PROP_SLICE]
        action_obs = obs[..., _ACTION_SLICE]
        task_obs = [obs[..., task_slice] for task_slice in _TASK_SLICES]

        prop_token = linen.Dense(self.embed_dim, name="prop_projection")(prop_obs)
        action_token = linen.Dense(self.embed_dim, name="action_projection")(action_obs)
        task_tokens = [
            linen.Dense(self.embed_dim, name=f"task_projection_{idx}")(token)
            for idx, token in enumerate(task_obs)
        ]
        task_tokens = jnp.stack(task_tokens, axis=-2)

        query_token = self.param(
            "query_token",
            linen.initializers.normal(stddev=0.02),
            (1, self.embed_dim),
        )
        query_token = jnp.broadcast_to(query_token, prop_token.shape[:-1] + query_token.shape)
        x = jnp.stack([prop_token, action_token], axis=-2)
        x = jnp.concatenate([x, query_token], axis=-2)

        for idx in range(self.num_layers):
            x = HumanoidTransformerBlock(
                self.embed_dim,
                self.num_heads,
                self.ff_dim,
                name=f"block_{idx}",
            )(x, task_tokens)

        x = RMSNorm(self.embed_dim, name="final_norm")(x)
        query_output = x[..., -1, :]
        return linen.Dense(
            self.output_dim,
            kernel_init=linen.initializers.zeros,
            bias_init=linen.initializers.zeros,
            name="projection_head",
        )(query_output)


def make_humanoid_transformer_ppo_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (512, 256, 128, 64),
    value_hidden_layer_sizes: Sequence[int] = (1024, 512, 256, 128),
    activation: Callable[[jax.Array], jax.Array] = linen.swish,
    policy_obs_key: str = "state",
    value_obs_key: str = "privileged_state",
    transformer_embed_dim: int = 256,
    transformer_num_heads: int = 4,
    transformer_ff_dim: int = 512,
    transformer_num_layers: int = 4,
    **unused_kwargs: Any,
) -> ppo_networks.PPONetworks:
    del policy_hidden_layer_sizes
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_module = HumanoidTransformerPolicy(
        output_dim=parametric_action_distribution.param_size,
        embed_dim=transformer_embed_dim,
        num_heads=transformer_num_heads,
        ff_dim=transformer_ff_dim,
        num_layers=transformer_num_layers,
    )

    def policy_apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        obs = obs if isinstance(obs, jax.Array) else obs[policy_obs_key]
        return policy_module.apply(policy_params, obs)

    policy_obs_size = _get_obs_state_size(observation_size, policy_obs_key)
    if policy_obs_size != 162:
        raise ValueError(
            f"humanoid_transformer requires {policy_obs_key} dim 162, got {policy_obs_size}"
        )
    dummy_policy_obs = jnp.zeros((1, policy_obs_size))
    policy_network = networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_policy_obs),
        apply=policy_apply,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
        activation=activation,
        obs_key=value_obs_key,
    )
    return ppo_networks.PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_network_factory(network_kind: str = "mlp", **kwargs: Any):
    if network_kind == "mlp":
        return ppo_networks.make_ppo_networks
    if network_kind == "humanoid_transformer":
        return make_humanoid_transformer_ppo_networks
    raise ValueError(f"Unsupported network_kind={network_kind!r}")
