"""DAgger distillation task for G1Cat."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict

import cat_ppo
from cat_ppo.envs.g1 import env_cat
from cat_ppo.envs.g1.env_cat import G1CatEnv


def g1_cat_dagger_task_config() -> config_dict.ConfigDict:
    config = env_cat.g1_loco_task_config()
    config.policy_config.dagger_config = config_dict.create(
        enable=True,
        loss="kl",
        teacher_restore_names=[],
        teacher_checkpoint_paths=[],
        kl_eps=1e-5,
    )
    config.env_config.pf_config.origin = [-0.5, -1.0, 0.0]
    return config


cat_ppo.registry.register("G1CatDagger", "config")(g1_cat_dagger_task_config())


@cat_ppo.registry.register("G1CatDagger", "train_env_class")
class G1CatDaggerEnv(G1CatEnv):
    """G1Cat with per-scene potential-field routing for DAgger teachers."""

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
                    "All G1CatDagger pf_config.paths must have matching sdf/bf/gf shapes; "
                    f"got mismatch at {pf_path}"
                )

        self.num_pf_scenes = len(pf_paths)
        self.sdf = jp.array(np.stack(sdf, axis=0))
        self.bf = jp.array(np.stack(bf, axis=0))
        self.gf = jp.array(np.stack(gf, axis=0))
        self.Nx, self.Ny, self.Nz, _ = self.sdf.shape[1:]
        self._field_pf_id = jp.array(0, dtype=jp.int32)

    def reset(self, rng: jax.Array):
        rng, pf_key = jax.random.split(rng)
        self._field_pf_id = jax.random.randint(pf_key, (), 0, self.num_pf_scenes)
        state = super().reset(rng)
        state.info["pf_id"] = self._field_pf_id
        return state

    def step(self, state, action):
        self._field_pf_id = state.info["pf_id"].astype(jp.int32)
        return super().step(state, action)

    def sample_field(self, field, pos):
        return super().sample_field(field[self._field_pf_id], pos)


cat_ppo.registry.register("G1CatDagger", "command_to_reference_fn")(env_cat.command_to_reference)
