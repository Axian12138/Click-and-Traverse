import functools

# --- Set environment variables ---
import os
from collections.abc import Mapping
from dataclasses import dataclass

import tyro
from absl import logging

os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# --- TensorFlow GPU setup ---
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.keras.mixed_precision.set_global_policy("float32")

import jax
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as rt
import tf2onnx
from mujoco_playground import wrapper
import cat_ppo

# --- MLP model definition ---
class MLP(tf.keras.Model):
    def __init__(
        self,
        layer_sizes,
        activation=tf.nn.relu,
        kernel_init="lecun_uniform",
        activate_final=False,
        bias=True,
        layer_norm=False,
    ):
        super().__init__()
        self.activation = activation
        self.activate_final = activate_final
        self.layer_norm = layer_norm
        self.model = tf.keras.Sequential(name="MLP_0")

        for i, size in enumerate(layer_sizes):
            self.model.add(
                tf.keras.layers.Dense(
                    size,
                    activation=None,
                    use_bias=bias,
                    kernel_initializer=kernel_init,
                    name=f"hidden_{i}",
                )
            )
            if i != len(layer_sizes) - 1 or activate_final:
                if layer_norm:
                    self.model.add(tf.keras.layers.LayerNormalization(name=f"ln_{i}"))

    def call(self, inputs):
        x = inputs
        for layer in self.model.layers:
            x = layer(x)
            if isinstance(layer, tf.keras.layers.Dense):
                if self.activate_final or not layer.name.endswith(
                    f"{len(self.model.layers) // (2 if self.layer_norm else 1) - 1}"
                ):
                    x = self.activation(x)
        loc, _ = tf.split(x, 2, axis=-1)
        return tf.tanh(loc)


class TFRMSNorm(tf.keras.layers.Layer):
    def __init__(self, dim, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.eps = eps

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(self.dim,),
            initializer="ones",
            trainable=True,
        )

    def call(self, x):
        norm = tf.linalg.norm(x, axis=-1, keepdims=True) / tf.sqrt(tf.cast(self.dim, x.dtype))
        return self.scale * x / (norm + self.eps)


class TFSwiGLU(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.keras.layers.Dense(hidden_dim, use_bias=False, name="w")
        self.v = tf.keras.layers.Dense(hidden_dim, use_bias=False, name="v")
        self.output_dense = None

    def build(self, input_shape):
        self.output_dense = tf.keras.layers.Dense(input_shape[-1], use_bias=False, name="output")
        super().build(input_shape)

    def call(self, x):
        return self.output_dense(tf.nn.silu(self.w(x)) * self.v(x))


class TFHumanoidTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        key_dim = embed_dim // num_heads
        self.rmsnorm1 = TFRMSNorm(embed_dim, name="rmsnorm1")
        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, output_shape=embed_dim, name="self_attention"
        )
        self.rmsnorm2 = TFRMSNorm(embed_dim, name="rmsnorm2")
        self.cond_norm = TFRMSNorm(embed_dim, name="cond_norm")
        self.cross_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, output_shape=embed_dim, name="cross_attention"
        )
        self.rmsnorm3 = TFRMSNorm(embed_dim, name="rmsnorm3")
        self.feed_forward = TFSwiGLU(ff_dim, name="feed_forward")
        self.self_mask = tf.constant(
            [[True, True, False], [True, True, False], [True, True, True]],
            dtype=tf.bool,
        )

    def call(self, x, task_tokens):
        x_norm = self.rmsnorm1(x)
        attn_output = self.self_attention(
            x_norm,
            x_norm,
            attention_mask=self.self_mask[None, :, :],
        )
        x = x + attn_output
        x_norm = self.rmsnorm2(x)
        task_norm = self.cond_norm(task_tokens)
        x = x + self.cross_attention(x_norm, task_norm)
        x = x + self.feed_forward(self.rmsnorm3(x))
        return x


class TFHumanoidTransformerPolicy(tf.keras.Model):
    def __init__(
        self,
        action_size,
        embed_dim=256,
        num_heads=4,
        ff_dim=512,
        num_layers=4,
    ):
        super().__init__()
        self.action_size = action_size
        self.embed_dim = embed_dim
        self.prop_projection = tf.keras.layers.Dense(embed_dim, name="prop_projection")
        self.action_projection = tf.keras.layers.Dense(embed_dim, name="action_projection")
        self.task_projections = [
            tf.keras.layers.Dense(embed_dim, name=f"task_projection_{idx}")
            for idx in range(8)
        ]
        self.blocks = [
            TFHumanoidTransformerBlock(embed_dim, num_heads, ff_dim, name=f"block_{idx}")
            for idx in range(num_layers)
        ]
        self.final_norm = TFRMSNorm(embed_dim, name="final_norm")
        self.projection_head = tf.keras.layers.Dense(action_size * 2, name="projection_head")

    def build(self, input_shape):
        self.query_token = self.add_weight(
            name="query_token",
            shape=(1, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def _slice_obs(self, obs):
        prop_obs = obs[:, 0:52]
        action_obs = obs[:, 52:76]
        task_obs = [
            obs[:, 76:85],
            obs[:, 85:92],
            obs[:, 92:99],
            obs[:, 99:106],
            obs[:, 106:120],
            obs[:, 120:134],
            obs[:, 134:148],
            obs[:, 148:162],
        ]
        return prop_obs, action_obs, task_obs

    def call(self, obs):
        prop_obs, action_obs, task_obs = self._slice_obs(obs)
        prop_token = self.prop_projection(prop_obs)
        action_token = self.action_projection(action_obs)
        task_tokens = [
            projection(token)
            for projection, token in zip(self.task_projections, task_obs)
        ]
        task_tokens = tf.stack(task_tokens, axis=1)
        batch_size = tf.shape(obs)[0]
        query_token = tf.broadcast_to(
            self.query_token[None, :, :],
            (batch_size, 1, self.embed_dim),
        )
        x = tf.stack([prop_token, action_token], axis=1)
        x = tf.concat([x, query_token], axis=1)
        for block in self.blocks:
            x = block(x, task_tokens)
        x = self.final_norm(x)
        logits = self.projection_head(x[:, -1, :])
        loc, _ = tf.split(logits, 2, axis=-1)
        return tf.tanh(loc)


# --- Utility functions ---
def build_tf_policy_network(
    action_size,
    hidden_layer_sizes,
    activation="swish",
    kernel_init="lecun_uniform",
    layer_norm=False,
):
    if activation == "swish":
        activation = tf.nn.swish
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

    return MLP(
        layer_sizes=list(hidden_layer_sizes) + [action_size * 2],
        activation=activation,
        kernel_init=kernel_init,
        layer_norm=layer_norm,
    )


def transfer_weights(jax_params, tf_model):
    for name, params in jax_params.items():
        try:
            tf_layer = tf_model.get_layer("MLP_0").get_layer(name=name)
        except ValueError:
            logging.error(f"Layer {name} not found in TF model.")
            continue
        if isinstance(tf_layer, tf.keras.layers.Dense):
            tf_layer.set_weights([np.array(params["kernel"]), np.array(params["bias"])])
        else:
            logging.error(f"Unhandled layer type: {type(tf_layer)}")
    logging.info("Weights transferred successfully.")


def _set_dense_weights(layer, params):
    weights = [np.array(params["kernel"])]
    if "bias" in params:
        weights.append(np.array(params["bias"]))
    layer.set_weights(weights)


def _set_rmsnorm_weights(layer, params):
    layer.set_weights([np.array(params["scale"])])


def _set_mha_weights(layer, params):
    layer.set_weights(
        [
            np.array(params["query"]["kernel"]),
            np.array(params["query"]["bias"]),
            np.array(params["key"]["kernel"]),
            np.array(params["key"]["bias"]),
            np.array(params["value"]["kernel"]),
            np.array(params["value"]["bias"]),
            np.array(params["out"]["kernel"]),
            np.array(params["out"]["bias"]),
        ]
    )


def _set_swiglu_weights(layer, params):
    _set_dense_weights(layer.w, params["w"])
    _set_dense_weights(layer.v, params["v"])
    _set_dense_weights(layer.output_dense, params["output"])


def _set_transformer_block_weights(layer, params):
    _set_rmsnorm_weights(layer.rmsnorm1, params["rmsnorm1"])
    _set_mha_weights(layer.self_attention, params["self_attention"])
    _set_rmsnorm_weights(layer.rmsnorm2, params["rmsnorm2"])
    _set_rmsnorm_weights(layer.cond_norm, params["cond_norm"])
    _set_mha_weights(layer.cross_attention, params["cross_attention"])
    _set_rmsnorm_weights(layer.rmsnorm3, params["rmsnorm3"])
    _set_swiglu_weights(layer.feed_forward, params["feed_forward"])


def transfer_transformer_weights(jax_params, tf_model):
    _set_dense_weights(tf_model.prop_projection, jax_params["prop_projection"])
    _set_dense_weights(tf_model.action_projection, jax_params["action_projection"])
    for idx, projection in enumerate(tf_model.task_projections):
        _set_dense_weights(projection, jax_params[f"task_projection_{idx}"])
    tf_model.query_token.assign(np.array(jax_params["query_token"]))
    for idx, block in enumerate(tf_model.blocks):
        _set_transformer_block_weights(block, jax_params[f"block_{idx}"])
    _set_rmsnorm_weights(tf_model.final_norm, jax_params["final_norm"])
    _set_dense_weights(tf_model.projection_head, jax_params["projection_head"])
    logging.info("Transformer weights transferred successfully.")


def get_latest_ckpt(path):
    from pathlib import Path

    ckpts = [ckpt for ckpt in Path(path).glob("*") if not ckpt.name.endswith(".json")]
    ckpts.sort(key=lambda x: int(x.name))
    return ckpts[-1] if ckpts else None


def _obs_shape(size):
    if isinstance(size, int):
        return (size,)
    return tuple(size)


def _normalize_obs_size(obs_size):
    if isinstance(obs_size, Mapping):
        return {key: _obs_shape(value) for key, value in obs_size.items()}

    shape = _obs_shape(obs_size)
    return {"state": shape, "privileged_state": shape}


def _policy_input_size(jax_params):
    policy_params = jax_params[1]["params"]
    if "hidden_0" in policy_params:
        return int(policy_params["hidden_0"]["kernel"].shape[0])
    if "prop_projection" in policy_params:
        return 162
    raise ValueError("Unable to infer policy input size from checkpoint params")


def convert_jax2onnx(
    ckpt_dir,
    output_path,
    inference_fn,
    hidden_layer_sizes,
    obs_size: int | Mapping[str, tuple[int, ...] | int],
    action_size: int,
    policy_obs_key,
    jax_params,
    network_kind="mlp",
    transformer_embed_dim=256,
    transformer_num_heads=4,
    transformer_ff_dim=512,
    transformer_num_layers=4,
    activation="swish",
):
    obs_size = _normalize_obs_size(obs_size)
    policy_input_size = _policy_input_size(jax_params)
    configured_policy_size = obs_size[policy_obs_key][0]
    logging.info(
        "ONNX export obs sizes: policy_obs_key=%s, state=%s, privileged_state=%s, "
        "checkpoint_policy_input=%d.",
        policy_obs_key,
        obs_size.get("state"),
        obs_size.get("privileged_state"),
        policy_input_size,
    )
    if configured_policy_size != policy_input_size:
        logging.warning(
            "Policy obs size mismatch during ONNX export: config/env has %s=%d, "
            "checkpoint policy expects %d. Using checkpoint shape for export.",
            policy_obs_key,
            configured_policy_size,
            policy_input_size,
        )
        obs_size[policy_obs_key] = (policy_input_size,)

    rand_obs = {
        "state": np.random.randn(1, obs_size["state"][0]).astype(np.float32),
        "privileged_state": np.random.randn(1, obs_size["privileged_state"][0]).astype(
            np.float32
        ),
    }

    jax_pred, _ = inference_fn(rand_obs, jax.random.PRNGKey(0))
    jax_pred = np.array(jax_pred[0])

    if network_kind == "mlp":
        tf_model = build_tf_policy_network(
            action_size=action_size,
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
        )
        example_input = tf.ones((1, policy_input_size))
        tf_model(example_input)  # build model
        transfer_weights(jax_params[1]["params"], tf_model)
    elif network_kind == "humanoid_transformer":
        if policy_input_size != 162:
            raise ValueError(
                f"humanoid_transformer ONNX export expects policy input 162, got {policy_input_size}"
            )
        tf_model = TFHumanoidTransformerPolicy(
            action_size=action_size,
            embed_dim=transformer_embed_dim,
            num_heads=transformer_num_heads,
            ff_dim=transformer_ff_dim,
            num_layers=transformer_num_layers,
        )
        example_input = tf.ones((1, policy_input_size))
        tf_model(example_input)  # build model
        transfer_transformer_weights(jax_params[1]["params"], tf_model)
    else:
        raise ValueError(f"Unsupported network_kind for ONNX export: {network_kind!r}")

    test_input = rand_obs[policy_obs_key].reshape(1, -1)
    tf_pred = tf_model(test_input)[0].numpy()
    max_abs_diff = float(np.max(np.abs(jax_pred - tf_pred)))
    logging.info("JAX/TF ONNX export check max_abs_diff=%g", max_abs_diff)
    if max_abs_diff > 1e-4:
        raise ValueError(f"JAX/TF output mismatch during ONNX export: {max_abs_diff}")

    tf_model.output_names = ["continuous_actions"]

    # Single input signature for ONNX conversion
    # spec = [
    #     tf.TensorSpec(
    #         shape=(1, obs_size[policy_obs_key][0]), dtype=tf.float32, name="obs"
    #     )
    # ]

    # Dynamic shape for ONNX conversion
    spec = (tf.TensorSpec([None, policy_input_size], tf.float32, name="obs"),)
    tf2onnx.convert.from_keras(
        tf_model, input_signature=spec, opset=11, output_path=output_path
    )

# --- CLI args ---
@dataclass
class Args:
    task: str
    exp_name: str


# --- Main entry point ---
def main(args: Args):
    import brax.training.agents.ppo.train as ppo
    from brax.training.agents.ppo.networks import make_ppo_networks
    from cat_ppo.learning.policy.ppo.humanoid_transformer_networks import (
        make_humanoid_transformer_ppo_networks,
    )

    import cat_ppo

    ckpt_path = cat_ppo.get_path_log(args.exp_name) / "checkpoints"
    latest_ckpt = get_latest_ckpt(ckpt_path)

    if latest_ckpt is None:
        raise FileNotFoundError("No checkpoint found.")

    logging.info(f"Using checkpoint: {latest_ckpt}")
    output_path = f"{latest_ckpt}/policy.onnx"

    env_class = cat_ppo.registry.get(args.task, "train_env_class")
    task_cfg = cat_ppo.registry.get(args.task, "config")
    env_cfg = task_cfg.env_config
    policy_config = task_cfg.policy_config
    env = env_class(task_type=env_cfg.task_type, config=env_cfg)

    policy_obs_key = policy_config.network_factory.policy_obs_key

    network_cfg = policy_config.network_factory.to_dict()
    network_kind = network_cfg.pop("network_kind", "mlp")
    if network_kind == "mlp":
        for key in (
            "transformer_embed_dim",
            "transformer_num_heads",
            "transformer_ff_dim",
            "transformer_num_layers",
        ):
            network_cfg.pop(key, None)
        network_fn = make_ppo_networks
    elif network_kind == "humanoid_transformer":
        network_fn = make_humanoid_transformer_ppo_networks
    else:
        raise ValueError(f"Unsupported network_kind={network_kind!r}")
    network_factory = functools.partial(network_fn, **network_cfg)
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=0,
        episode_length=policy_config.episode_length,
        normalize_observations=False,
        restore_checkpoint_path=latest_ckpt,
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_envs=1,
    )

    make_inference_fn, params, _ = train_fn(environment=env)
    inference_fn = make_inference_fn(params, deterministic=True)

    obs_size = env.observation_size
    act_size = env.action_size

    convert_jax2onnx(
        ckpt_dir=latest_ckpt,
        output_path=output_path,
        inference_fn=inference_fn,
        hidden_layer_sizes=policy_config.network_factory.policy_hidden_layer_sizes,
        obs_size=obs_size,
        action_size=act_size,
        policy_obs_key=policy_obs_key,
        jax_params=params,
        network_kind=network_kind,
        transformer_embed_dim=policy_config.network_factory.get("transformer_embed_dim", 256),
        transformer_num_heads=policy_config.network_factory.get("transformer_num_heads", 4),
        transformer_ff_dim=policy_config.network_factory.get("transformer_ff_dim", 512),
        transformer_num_layers=policy_config.network_factory.get("transformer_num_layers", 4),
        activation="swish",
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
