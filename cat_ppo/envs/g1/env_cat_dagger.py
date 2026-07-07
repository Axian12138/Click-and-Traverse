"""DAgger distillation tasks for G1Cat variants."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict

import cat_ppo
from cat_ppo.envs.g1 import env_cat
from cat_ppo.envs.g1.env_cat import G1CatEnv


def _add_dagger_config(config: config_dict.ConfigDict) -> config_dict.ConfigDict:
    config.policy_config.network_factory.policy_hidden_layer_sizes = (512, 256, 128, 64)
    config.policy_config.network_factory.value_hidden_layer_sizes = (1024, 512, 256, 128)
    config.policy_config.entropy_cost = 0.003
    config.policy_config.dagger_config = config_dict.create(
        enable=True,
        loss="kl",
        teacher_restore_names=[],
        teacher_checkpoint_paths=[],
        kl_eps=1e-5,
        dagger_timesteps=0,
        actor_loss_scale=1.0,
        value_loss_scale=2.0,
        adaptive_pf_sampling=True,
        pf_sampling_alpha=1.0,
        pf_sampling_ema_decay=0.95,
    )
    config.env_config.pf_config.origin = [-0.5, -1.0, 0.0]
    config.env_config.pf_config.sampling_weights = []
    config.env_config.pf_config.sampling_alpha = 1.0
    config.env_config.pf_config.sampling_ema_decay = 0.95
    # V8
    config.policy_config.discounting = 0.98 
    config.policy_config.unroll_length = 32
    config.policy_config.num_envs = 32768*2
    config.policy_config.batch_size = 2048
    config.policy_config.num_minibatches = 32*2
    return config


def g1_cat_dagger_task_config() -> config_dict.ConfigDict:
    return _add_dagger_config(env_cat.g1_loco_task_config())



def _load_pf_fields(config: config_dict.ConfigDict):
    pf_paths = list(getattr(config.pf_config, "paths", []))
    if not pf_paths:
        pf_paths = [config.pf_config.path]

    sdf, bf, gf = [], [], []
    for pf_path in pf_paths:
        sdf.append(np.load(f"{pf_path}/sdf.npy")[..., None])
        bf.append(np.load(f"{pf_path}/bf.npy"))
        gf.append(np.load(f"{pf_path}/gf.npy"))

    sdf_shape, bf_shape, gf_shape = sdf[0].shape, bf[0].shape, gf[0].shape
    for pf_path, sdf_arr, bf_arr, gf_arr in zip(pf_paths, sdf, bf, gf):
        if sdf_arr.shape != sdf_shape or bf_arr.shape != bf_shape or gf_arr.shape != gf_shape:
            raise ValueError(
                "All DAgger pf_config.paths must have matching sdf/bf/gf shapes; "
                f"got mismatch at {pf_path}"
            )

    sampling_weights = list(getattr(config.pf_config, "sampling_weights", []))
    if not sampling_weights:
        sampling_weights = [1.0] * len(pf_paths)
    if len(sampling_weights) != len(pf_paths):
        raise ValueError(
            f"pf_config.sampling_weights length must match pf paths: "
            f"{len(sampling_weights)} != {len(pf_paths)}"
        )
    sampling_weights = np.asarray(sampling_weights, dtype=np.float32)
    if np.any(sampling_weights < 0.0) or not np.any(sampling_weights > 0.0):
        raise ValueError("pf_config.sampling_weights must be non-negative with at least one positive value")

    return pf_paths, sdf, bf, gf, sampling_weights


cat_ppo.registry.register("G1CatDagger", "config")(g1_cat_dagger_task_config())


class _DaggerSceneMixin:
    """Per-scene potential-field routing shared by G1Cat and G1CatPri DAgger."""

    def __init__(
        self,
        task_type: str = "flat_terrain",
        config: config_dict.ConfigDict = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            task_type=task_type,
            config=config,
            config_overrides=config_overrides,
        )
        _, sdf, bf, gf, sampling_weights = _load_pf_fields(config)

        self.num_pf_scenes = len(sdf)
        self.sdf = jp.array(np.stack(sdf, axis=0))
        self.bf = jp.array(np.stack(bf, axis=0))
        self.gf = jp.array(np.stack(gf, axis=0))
        self.Nx, self.Ny, self.Nz, _ = self.sdf.shape[1:]
        self._field_pf_id = jp.array(0, dtype=jp.int32)
        sampling_weights = sampling_weights / sampling_weights.sum()
        self._pf_sampling_logits = jp.log(jp.array(sampling_weights) + 1e-8)
        self._pf_sampling_alpha = float(getattr(config.pf_config, "sampling_alpha", 1.0))
        self._pf_sampling_ema_decay = float(getattr(config.pf_config, "sampling_ema_decay", 0.95))

    def reset(self, rng: jax.Array):
        rng, pf_key = jax.random.split(rng)
        self._field_pf_id = jax.random.categorical(pf_key, self._pf_sampling_logits).astype(jp.int32)
        state = super().reset(rng)
        state.info["pf_id"] = self._field_pf_id
        state.info["pf_success_ema"] = jp.zeros(self.num_pf_scenes)
        state.info["pf_episode_ema"] = jp.zeros(self.num_pf_scenes)
        state.info["pf_sampling_logits"] = self._pf_sampling_logits
        state.info["pf_sampling_alpha"] = jp.array(self._pf_sampling_alpha, dtype=jp.float32)
        state.info["pf_sampling_ema_decay"] = jp.array(self._pf_sampling_ema_decay, dtype=jp.float32)
        return state

    def reset_with_pf_id(self, rng: jax.Array, pf_id: jax.Array):
        self._field_pf_id = pf_id.astype(jp.int32)
        state = super().reset(rng)
        state.info["pf_id"] = self._field_pf_id
        state.info["pf_success_ema"] = jp.zeros(self.num_pf_scenes)
        state.info["pf_episode_ema"] = jp.zeros(self.num_pf_scenes)
        state.info["pf_sampling_logits"] = self._pf_sampling_logits
        state.info["pf_sampling_alpha"] = jp.array(self._pf_sampling_alpha, dtype=jp.float32)
        state.info["pf_sampling_ema_decay"] = jp.array(self._pf_sampling_ema_decay, dtype=jp.float32)
        return state

    def step(self, state, action):
        self._field_pf_id = state.info["pf_id"].astype(jp.int32)
        return super().step(state, action)

    def sample_field(self, field, pos):
        return super().sample_field(field[self._field_pf_id], pos)


@cat_ppo.registry.register("G1CatDagger", "train_env_class")
class G1CatDaggerEnv(_DaggerSceneMixin, G1CatEnv):
    """G1Cat with per-scene potential-field routing for DAgger teachers."""



cat_ppo.registry.register("G1CatDagger", "command_to_reference_fn")(env_cat.command_to_reference)
