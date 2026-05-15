"""Long-track variant of G1Cat."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

import cat_ppo
from cat_ppo.envs.g1 import env_cat
from cat_ppo.envs.g1.env_cat import G1CatEnv


def g1_cat_long_task_config() -> config_dict.ConfigDict:
    config = env_cat.g1_loco_task_config()
    reward_scales = config.env_config.reward_config.scales
    reward_scales.stage_progress = 1.0
    reward_scales.idle_default_pos = 0.5

    config.env_config.stage_reward_interval = 1.0
    config.env_config.scene_bounds_margin = 0.0
    return config


cat_ppo.registry.register("G1CatLong", "config")(g1_cat_long_task_config())


@cat_ppo.registry.register("G1CatLong", "train_env_class")
class G1CatLongEnv(G1CatEnv):
    """G1Cat with dense progress rewards and scene-bound termination."""

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
        self._scene_xy_min = self.pf_origin[:2]
        self._scene_xy_max = self.pf_origin[:2] + jp.array([self.Nx, self.Ny]) * self.dx

    def reset(self, rng: jax.Array):
        state = super().reset(rng)
        state.info["stage_idx"] = self._stage_index(state.data.qpos[0])
        return state

    def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        done = super()._get_termination(data, info)
        margin = self._config.scene_bounds_margin
        root_xy = data.qpos[:2]
        out_of_scene = jp.any(root_xy < (self._scene_xy_min - margin))
        out_of_scene |= jp.any(root_xy > (self._scene_xy_max + margin))
        return done | out_of_scene

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        done: jax.Array,
        feet_contact: jax.Array,
    ) -> dict[str, jax.Array]:
        reward_dict = super()._get_reward(data, action, info, done, feet_contact)
        reward_dict["stage_progress"] = self._reward_stage_progress(data.qpos[0], info)
        reward_dict["idle_default_pos"] = self._reward_idle_default_pos(data.qpos[7:], info["command"][0])
        return reward_dict

    def _stage_index(self, root_x: jax.Array) -> jax.Array:
        progress_x = root_x - self.pf_origin[0]
        stage_idx = jp.floor(progress_x / self._config.stage_reward_interval)
        return jp.maximum(stage_idx.astype(jp.int32), 0)

    def _reward_stage_progress(self, root_x: jax.Array, info: dict[str, Any]) -> jax.Array:
        prev_stage = info["stage_idx"]
        current_stage = self._stage_index(root_x)
        reached_stage = jp.maximum(prev_stage, current_stage)
        stage_delta = jp.maximum(reached_stage - prev_stage, 0)
        info["stage_idx"] = reached_stage
        return stage_delta.astype(jp.float32) / self.dt

    def _reward_idle_default_pos(self, joint_pos: jax.Array, move_flag: jax.Array) -> jax.Array:
        joint_err = jp.mean(jp.square(joint_pos - self._default_qpos))
        reward = jp.exp(-10.0 * joint_err)
        return jp.where(move_flag < 0.5, reward, 0.0)


cat_ppo.registry.register("G1CatLong", "command_to_reference_fn")(env_cat.command_to_reference)
