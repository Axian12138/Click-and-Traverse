# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Unitree G1."""

from typing import Any, Dict, Optional, Union
import jax
import jaxlie
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np
from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

import cat_ppo
from cat_ppo.envs.g1.env_loco import G1LocoEnv
from cat_ppo.envs.g1 import constants as consts

ENABLE_RANDOMIZE = True


def g1_loco_task_config() -> config_dict.ConfigDict:
    from cat_ppo.envs.g1.randomize import domain_randomize

    env_config = config_dict.create(
        task_type="flat_terrain",
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.5,
        history_len=15,
        num_obs=85,
        num_pri=108,
        num_act=12,
        restricted_joint_range=False,
        soft_joint_pos_limit_factor=0.95,
        gait_config=config_dict.create(
            gait_bound=0.6,
            freq_range=[1.3, 1.5],
            foot_height_range=[0.05, 0.05],
        ),
        dm_rand_config=config_dict.create(
            enable_pd=True,
            kp_range=[0.75, 1.25],
            kd_range=[0.75, 1.25],
            enable_rfi=True,
            rfi_lim=0.1,
            rfi_lim_range=[0.5, 1.5],
            enable_ctrl_delay=False,
            ctrl_delay_range=[0, 2],
        ),
        noise_config=config_dict.create(
            level=1.0,
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gravity=0.05,
                gyro=0.2,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_orientation=2.0,
                tracking_lin_vel=1.0,
                tracking_ang_vel=0.75,
                body_motion=-2.0,
                body_rotation=1.0,
                feet_rotation=0.1,
                foot_contact=-1.0,
                foot_clearance=-15.0,
                foot_slip=-0.1,
                foot_balance=-10,
                straight_knee=-30,
                smoothness_joint=-1e-5,
                smoothness_action=-1e-2,
                joint_limits=-1.0,
                joint_torque=-1e-4,
            ),
            base_height_target=0.75,
            foot_height_stance=0.0,
        ),
        push_config=config_dict.create(
            enable=True,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        command_config=config_dict.create(
            resampling_time=10.0,
            stop_prob=0.2,
        ),
        lin_vel_x=[-0.7, 0.7],
        lin_vel_y=[-0.7, 0.7],
        ang_vel_yaw=[-0.7, 0.7],
        torso_height=[0.5, consts.DEFAULT_CHEST_Z],
    )

    policy_config = config_dict.create(
        num_timesteps=5_000_000_000,
        max_devices_per_host=8,
        wrap_env=True,
        madrona_backend=False,
        augment_pixels=False,
        num_envs=32768,
        episode_length=1000,
        action_repeat=1,
        wrap_env_fn=None,
        randomization_fn=domain_randomize if ENABLE_RANDOMIZE else None,
        learning_rate=3e-4,
        entropy_cost=0.01,
        discounting=0.97,
        unroll_length=20,
        batch_size=1024,
        num_minibatches=32,
        num_updates_per_batch=4,
        num_resets_per_eval=0,
        normalize_observations=False,
        reward_scaling=1.0,
        clipping_epsilon=0.2,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        normalize_advantage=True,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(256, 128, 64),
            value_hidden_layer_sizes=(512, 256, 128),
            policy_obs_key="state",
            value_obs_key="privileged_state",
        ),
        seed=0,
        num_evals=6,
        eval_env=None,
        num_eval_envs=0,
        deterministic_eval=False,
        log_training_metrics=True,
        training_metrics_steps=int(1e6),
        progress_fn=lambda *args: None,
        save_checkpoint_path=None,
        restore_checkpoint_path=None,
        restore_params=None,
        restore_value_fn=False,
    )

    eval_config = config_dict.create(
        duration=50.0,
        command_waypoints=np.array(
            [
                [0, 0.0, 0.0, 0.0],
            ]
        ),
    )

    config = config_dict.create(
        env_config=env_config,
        policy_config=policy_config,
        eval_config=eval_config,
    )
    return config

cat_ppo.registry.register("G1CatLoco", "config")(g1_loco_task_config())

def base2navi_transform(base2world: jax.Array) -> jax.Array:
    x = base2world[:, 0]
    x_proj = x.at[2].set(0.0)
    x_proj /= jp.linalg.norm(x_proj)
    z_axis = jp.array([0.0, 0.0, 1.0])
    y_axis = jp.cross(z_axis, x_proj)
    y_axis /= jp.linalg.norm(y_axis)
    x_axis = jp.cross(y_axis, z_axis)
    return jp.column_stack((x_axis, y_axis, z_axis))


def torque_step(
        rng: jax.Array,
        model: mjx.Model,
        data: mjx.Data,
        qpos_des: jax.Array,
        kps: jax.Array,
        kds: jax.Array,
        kp_scale: jax.Array,
        kd_scale: jax.Array,
        rfi_lim_scale: jax.Array,
        torque_limit: jax.Array,
        n_substeps: int = 1,
) -> tuple[jax.Array, mjx.Data]:
    def single_step(carry, _):
        rng, data = carry
        rng, rng_rfi = jax.random.split(rng, 2)

        pos_err = qpos_des - data.qpos[7:]
        vel_err = -data.qvel[6:]
        torque = (kp_scale * kps) * pos_err + (kd_scale * kds) * vel_err

        rfi_noise = rfi_lim_scale * jax.random.uniform(rng_rfi, shape=torque.shape, minval=-1.0, maxval=1.0)
        torque += rfi_noise

        torque = jp.clip(torque, -torque_limit, torque_limit)

        data = data.replace(ctrl=torque)
        data = mjx.step(model, data)

        return (rng, data), None

    return jax.lax.scan(single_step, (rng, data), (), n_substeps)[0]


@cat_ppo.registry.register("G1CatLoco", "train_env_class")
class G1CatLocoEnv(G1LocoEnv):
    """Track a joystick command."""

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
        self._head_site_id = self._mj_model.site("head").id

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q.copy()
        qvel = jp.zeros(self.mjx_model.nv)

        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
        qpos = qpos.at[2].set(0.8)

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-np.pi / 2, maxval=np.pi / 2)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)

        rng, key = jax.random.split(rng)
        rand_qpos = qpos[7:] * jax.random.uniform(key, (29,), minval=0.5, maxval=1.5)
        rand_qpos = jp.clip(rand_qpos, self._soft_lowers, self._soft_uppers)
        qpos = qpos.at[7:].set(rand_qpos)

        rng, key = jax.random.split(rng)
        qvel = qvel.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

        head_pos = data.site_xpos[self._head_site_id]
        head_vel = jp.zeros_like(head_pos)
        feet_pos = data.site_xpos[self._feet_site_id]
        feet_vel = jp.zeros_like(feet_pos)
        hands_pos = data.site_xpos[self._hands_site_id]
        hands_vel = jp.zeros_like(hands_pos)
        rng, cmd_rng = jax.random.split(rng)
        command = self.sample_command(cmd_rng)

        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        rng, gait_freq_rng, foot_height_rng = jax.random.split(rng, 3)
        gait_freq = jax.random.uniform(
            gait_freq_rng,
            minval=self._config.gait_config.freq_range[0],
            maxval=self._config.gait_config.freq_range[1],
        )
        phase_dt = 2 * jp.pi * self.dt * gait_freq
        rng, phase_rng = jax.random.split(rng)
        cond_phase = jax.random.bernoulli(phase_rng)
        phase = jp.where(cond_phase, self._init_phase_l, self._init_phase_r)
        foot_height = jax.random.uniform(
            foot_height_rng,
            minval=self._config.gait_config.foot_height_range[0],
            maxval=self._config.gait_config.foot_height_range[1],
        )

        rng, key_kp, key_kd, key_rfi = jax.random.split(rng, 4)
        kp_scale = jax.random.uniform(
            key_kp,
            minval=self._config.dm_rand_config.kp_range[0],
            maxval=self._config.dm_rand_config.kp_range[1],
        )
        kp_scale = jp.where(self._config.dm_rand_config.enable_pd, kp_scale, jp.ones_like(kp_scale))

        kd_scale = jax.random.uniform(
            key_kd,
            minval=self._config.dm_rand_config.kd_range[0],
            maxval=self._config.dm_rand_config.kd_range[1],
        )
        kd_scale = jp.where(self._config.dm_rand_config.enable_pd, kd_scale, jp.ones_like(kd_scale))

        rfi_lim_noise_scale = jax.random.uniform(
            key_rfi,
            self.torque_limit.shape,
            minval=self._config.dm_rand_config.rfi_lim_range[0],
            maxval=self._config.dm_rand_config.rfi_lim_range[1],
        )
        rfi_lim_scale = self._config.dm_rand_config.rfi_lim * rfi_lim_noise_scale * self.torque_limit
        rfi_lim_scale = jp.where(self._config.dm_rand_config.enable_rfi, rfi_lim_scale, jp.zeros_like(rfi_lim_scale))

        info = {
            "rng": rng,
            "step": 0,
            "command": command,
            "last_command": jp.zeros(4),
            "last_act": jp.zeros(self.action_size),
            "last_last_act": jp.zeros(self.action_size),
            "last_feet_vel": jp.zeros(2),
            "last_joint_vel": np.zeros(self.num_joints),
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            "motor_targets": self._default_qpos.copy(),
            "navi2world_rot": jp.eye(3),
            "navi2world_pose": jp.eye(4),
            "navi_torso_rpy": jp.zeros(3),
            "navi_torso_lin_vel": jp.zeros(3),
            "navi_torso_ang_vel": jp.zeros(3),
            "navi_pelvis_rpy": jp.zeros(3),
            "navi_pelvis_lin_vel": jp.zeros(3),
            "navi_pelvis_ang_vel": jp.zeros(3),
            "phase": phase,
            "phase_dt": phase_dt,
            "gait_mask": jp.zeros(2),
            "gait_freq": gait_freq,
            "foot_height": foot_height,
            "stop_timestep": 100,
            "kp_scale": kp_scale,
            "kd_scale": kd_scale,
            "rfi_lim_scale": rfi_lim_scale,
            "head_pos": head_pos.copy(),
            "head_vel": head_vel.copy(),
            "feet_pos": feet_pos.copy(),
            "feet_vel": feet_vel.copy(),
            "hands_pos": hands_pos.copy(),
            "hands_vel": hands_vel.copy(),
        }
        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state.info["rng"], push1_rng, push2_rng = jax.random.split(state.info["rng"], 3)

        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push_signal = jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= push_signal
        push *= self._config.push_config.enable

        qvel = state.data.qvel
        qvel = qvel.at[:2].set(qvel[:2] + push * push_magnitude)
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        lower_motor_targets = jp.clip(
            state.info["motor_targets"][self.action_joint_ids]
            + action * self._config.action_scale,
            self._soft_lowers[self.action_joint_ids],
            self._soft_uppers[self.action_joint_ids],
        )
        motor_targets = self._default_qpos.copy()
        motor_targets = motor_targets.at[self.action_joint_ids].set(lower_motor_targets)

        state.info["rng"], data = torque_step(
            state.info["rng"],
            self.mjx_model,
            state.data,
            motor_targets,
            kps=self._kps,
            kds=self._kds,
            kp_scale=state.info["kp_scale"],
            kd_scale=state.info["kd_scale"],
            rfi_lim_scale=state.info["rfi_lim_scale"],
            torque_limit=self.torque_limit,
            n_substeps=self.n_substeps,
        )

        feet_contact = jp.array([geoms_colliding(data, geom_id, self._floor_geom_id) for geom_id in self._feet_geom_id])
        state.info["motor_targets"] = motor_targets

        pelvis2world_rot = data.site_xmat[self._pelvis_imu_site_id]
        navi2world_rot = base2navi_transform(pelvis2world_rot)
        state.info["navi2world_pose"] = state.info["navi2world_pose"].at[:3, :3].set(navi2world_rot)
        state.info["navi2world_pose"] = (
            state.info["navi2world_pose"].at[:2, 3].set(data.site_xpos[self._pelvis_imu_site_id][:2])
        )
        state.info["navi2world_pose"] = (
            state.info["navi2world_pose"].at[2, 3].set(self._config.reward_config.base_height_target)
        )

        pelvis2navi_rot = navi2world_rot.T @ pelvis2world_rot
        state.info["navi2world_rot"] = navi2world_rot
        state.info["navi_pelvis_rpy"] = jp.array(jaxlie.SO3.from_matrix(pelvis2navi_rot).as_rpy_radians())
        state.info["navi_pelvis_lin_vel"] = pelvis2navi_rot @ self.get_local_linvel(data, "pelvis")
        state.info["navi_pelvis_ang_vel"] = pelvis2navi_rot @ self.get_gyro(data, "pelvis")
        torso2world_rot = data.site_xmat[self._torso_imu_site_id]
        torso2navi_rot = navi2world_rot.T @ torso2world_rot
        state.info["navi_torso_rpy"] = jp.array(jaxlie.SO3.from_matrix(torso2navi_rot).as_rpy_radians())
        state.info["navi_torso_lin_vel"] = torso2navi_rot @ self.get_local_linvel(data, "torso")
        state.info["navi_torso_ang_vel"] = torso2navi_rot @ self.get_gyro(data, "torso")

        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])

        state.info["last_command"] = state.info["command"].copy()
        state.info["command"] = jp.where(
            state.info["step"] % self._cmd_resample_steps == 0,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        head_pos = data.site_xpos[self._head_site_id]
        head_vel = (head_pos - state.info["head_pos"]) / self.dt
        feet_pos = data.site_xpos[self._feet_site_id]
        feet_vel = (feet_pos - state.info["feet_pos"]) / self.dt
        hands_pos = data.site_xpos[self._hands_site_id]
        hands_vel = (hands_pos - state.info["hands_pos"]) / self.dt

        state.info["head_pos"] = head_pos.copy()
        state.info["head_vel"] = head_vel.copy()
        state.info["feet_pos"] = feet_pos.copy()
        state.info["feet_vel"] = feet_vel.copy()
        state.info["hands_pos"] = hands_pos.copy()
        state.info["hands_vel"] = hands_vel.copy()
        state.info["push"] = push
        state.info["push_step"] += 1
        state.info["step"] += 1

        self._update_phase(state)

        state.info["last_last_act"] = state.info["last_act"].copy()
        state.info["last_act"] = action.copy()
        obs = self._get_obs(data, state.info, feet_contact)
        done = self._get_termination(data, state.info)

        rewards = self._get_reward(data, action, state.info, done, feet_contact)
        rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)


        timeout = state.info["step"] >= self._config.episode_length
        state.info["step"] = jp.where(done | timeout, 0, state.info["step"])

        state.info["motor_targets"] = jp.where(
            done, self._default_qpos, state.info["motor_targets"]
        )
        state.info["rng"], episode_rng = jax.random.split(state.info["rng"])
        _is_resample = jp.where(
            done,
            self.resample_domain_random_param(episode_rng, state),
            False,
        )

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        state.info["last_joint_vel"] = data.qvel[6:].copy()
        state.info["last_feet_vel"] = data.sensordata[self._foot_linvel_sensor_adr][..., 2]
        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state


    def sample_command(self, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        lin_vel_x = jax.random.uniform(rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1])
        lin_vel_y = jax.random.uniform(rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1])
        ang_vel_yaw = jax.random.uniform(rng3, minval=self._config.ang_vel_yaw[0], maxval=self._config.ang_vel_yaw[1])

        command = jp.hstack([1.0, lin_vel_x, lin_vel_y, ang_vel_yaw])

        small_cond = jp.linalg.norm(command[1:4]) < 0.2
        command = jp.where(small_cond, self._stop_cmd, command)

        stop_cond = jax.random.bernoulli(rng4, p=self._cmd_stop_prob)
        command = jp.where(stop_cond, self._stop_cmd, command)

        return command


    def _get_termination(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        fall_termination = self.get_gravity(data, "pelvis")[2] < 0.0
        fall_termination |= info["head_pos"][2] < 0.7
        contact_termination = collision.geoms_colliding(
            data,
            self._right_foot_geom_id,
            self._left_foot_geom_id,
        )
        contact_termination |= collision.geoms_colliding(
            data,
            self._left_foot_geom_id,
            self._right_shin_geom_id,
        )
        contact_termination |= collision.geoms_colliding(
            data,
            self._right_foot_geom_id,
            self._left_shin_geom_id,
        )
        return fall_termination | contact_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    def _get_obs(self, data: mjx.Data, info: dict[str, Any], feet_contact: jax.Array) -> mjx_env.Observation:
        gyro_pelvis = self.get_gyro(data, "pelvis")
        gvec_pelvis = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
        linvel_pelvis = self.get_local_linvel(data, "pelvis")
        joint_angles = data.qpos[7:]
        joint_vel = data.qvel[6:]

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro_pelvis = (
                gyro_pelvis
                + (2 * jax.random.uniform(noise_rng, shape=gyro_pelvis.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.gyro
        )

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gvec_pelvis = (
                gvec_pelvis
                + (2 * jax.random.uniform(noise_rng, shape=gvec_pelvis.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.gravity
        )

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
                joint_angles
                + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.joint_pos
        )

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
                joint_vel
                + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
                * self._config.noise_config.level
                * self._config.noise_config.scales.joint_vel
        )
        gait_phase = jp.concatenate([jp.cos(info["phase"]), jp.sin(info["phase"])])
        state = jp.hstack(
            [
                noisy_gyro_pelvis,
                noisy_gvec_pelvis,
                (noisy_joint_angles - self._default_qpos)[self.obs_joint_ids],
                noisy_joint_vel[self.obs_joint_ids],
                info["last_act"],
                info["motor_targets"][self.action_joint_ids],
                info["command"],
                info["foot_height"],
                gait_phase,
            ]
        )
        privileged_state = jp.hstack(
            [
                gyro_pelvis,
                gvec_pelvis,
                linvel_pelvis,
                (joint_angles - self._default_qpos)[self.obs_joint_ids],
                joint_vel[self.obs_joint_ids],
                info["last_act"],
                info["motor_targets"][self.action_joint_ids],
                info["command"],
                info["foot_height"],
                gait_phase,
                info["navi_torso_rpy"][:2],
                info["gait_mask"],
                feet_contact,
                info["kp_scale"],
                info["kd_scale"],
                info["rfi_lim_scale"],
            ]
        )

        state = jp.nan_to_num(state)
        privileged_state = jp.nan_to_num(privileged_state)

        return {"state": state, "privileged_state": privileged_state}

    def _get_reward(
            self,
            data: mjx.Data,
            action: jax.Array,
            info: dict[str, Any],
            done: jax.Array,
            feet_contact: jax.Array,
    ) -> dict[str, jax.Array]:
        move_flag = info["command"][0]
        cmd_vel = info["command"][1:].copy()

        reward_dict = {
            "tracking_orientation": self._reward_orientation(
                info["navi_pelvis_rpy"], info["navi_torso_rpy"], info["head_pos"][2] > (self._config.torso_height[1] + 0.1)
            ),
            "tracking_lin_vel": self._reward_tracking_lin_vel(cmd_vel, info["navi_pelvis_lin_vel"]),
            "tracking_ang_vel": self._reward_tracking_ang_vel(cmd_vel, info["navi_pelvis_ang_vel"]),
            "body_motion": self._cost_body_motion(info["navi_pelvis_lin_vel"], info["navi_torso_ang_vel"], cmd_vel),
            "body_rotation": self._reward_body_rotation(data, cmd_vel, info["navi2world_rot"]),
            "feet_rotation": self._reward_feet_rotation(data, info["navi2world_rot"]),
            "foot_contact": self._cost_foot_contact(data, feet_contact, info["gait_mask"], move_flag),
            "foot_clearance": self._cost_foot_clearance(data, info["foot_height"], info["gait_mask"], move_flag),
            "foot_slip": self._cost_foot_slip(data, info["gait_mask"]),
            "foot_balance": self._cost_foot_balance(data, info["navi2world_pose"], move_flag),
            "straight_knee": self._cost_straight_knee(data.qpos[jp.array(self._knee_indices) + 7]),
            "joint_limits": self._cost_joint_pos_limits(data.qpos[7:]),
            "joint_torque": self._cost_torque(data.actuator_force),
            "smoothness_joint": self._cost_smoothness_joint(data, info["last_joint_vel"]),
            "smoothness_action": self._cost_smoothness_action(action, info["last_act"], info["last_last_act"]),
        }
        for k, v in reward_dict.items():
            reward_dict[k] = jp.where(jp.isnan(v), 0.0, v)

        return reward_dict

    def _cost_body_motion(
        self, local_lin_vel, local_ang_vel: jax.Array, cmd_vel: jax.Array
    ) -> jax.Array:
        cmd_xy = cmd_vel[:2]
        cmd_norm = jp.linalg.norm(cmd_xy)
        is_zero_cmd = jp.isclose(cmd_norm, 0.0)
        cmd_dir = jp.where(is_zero_cmd, jp.zeros_like(cmd_xy), cmd_xy / cmd_norm)

        lin_xy = local_lin_vel[:2]
        lin_xy_orth = lin_xy - jp.dot(lin_xy, cmd_dir) * cmd_dir
        cost_lin_xy_orth = jp.where(is_zero_cmd, 0.0, jp.sum(jp.square(lin_xy_orth)))

        cost = (
            1.2 * cost_lin_xy_orth
            + 0.4 * jp.abs(local_ang_vel[0])
            + 0.4 * jp.abs(local_ang_vel[1])
        )
        return cost

    def _reward_feet_rotation(
        self, data: mjx.Data, navi2world_rot: jax.Array
    ) -> jax.Array:
        knees2world_rot = jp.concat(
            [
                data.xmat[self.body_id_knee_l][None],
                data.xmat[self.body_id_knee_r][None],
            ]
        )
        knees2navi_rot = navi2world_rot.T[None] @ knees2world_rot
        ankles2world_rot = jp.concat(
            [
                data.xmat[self.body_id_ankle_l][None],
                data.xmat[self.body_id_ankle_r][None],
            ]
        )
        ankles2navi_rot = navi2world_rot.T[None] @ ankles2world_rot

        knees_roll_err = jp.sum(jp.abs(knees2navi_rot[:, 2, 1]))
        knees_yaw_err = jp.sum(jp.abs(knees2navi_rot[:, 0, 1]))
        ankles_roll_err = jp.sum(jp.abs(ankles2navi_rot[:, 1, 2]))
        ankles_pitch_err = jp.sum(jp.abs(ankles2navi_rot[:, 0, 2]))
        ankles_yaw_err = jp.sum(jp.square(ankles2navi_rot[:, 0, 1]))

        axis_rew = jp.exp(
            -1.0
            * (
                knees_roll_err
                + knees_yaw_err
                + ankles_roll_err
                + ankles_pitch_err
                + ankles_yaw_err
            )
        )
        return axis_rew

    def _reward_orientation(
        self, pelvis_rpy: jax.Array, torso_rpy: jax.Array, idle_mask: jax.Array
    ) -> jax.Array:
        err_roll = jp.abs(pelvis_rpy[0]) + jp.abs(torso_rpy[0])
        err_pitch_dire = jp.abs(jp.clip(torso_rpy[1], -np.pi, 0.0))
        err_pitch_idle = idle_mask * jp.abs(torso_rpy[1])
        err_ori = err_roll + err_pitch_dire + err_pitch_idle
        rew = jp.exp(-0.5 * err_ori) - err_pitch_dire
        return rew

    def _cost_straight_knee(self, knee_pos) -> jax.Array:
        penalty = jp.clip(0.1 - knee_pos, min=0.0)
        cost = jp.sum(penalty)
        return cost

    def _cost_foot_balance(
        self, data: mjx.Data, navi2world_pose: jax.Array, task_mask: jax.Array
    ):
        stance_mask = 1 - task_mask
        sup2world_pos_h = jp.ones((3, 4))
        sup2world_pos_h = sup2world_pos_h.at[0, :3].set(
            data.subtree_com[self.body_id_pelvis]
        )
        sup2world_pos_h = sup2world_pos_h.at[1, :3].set(
            data.site_xpos[self._feet_site_id[0]]
        )
        sup2world_pos_h = sup2world_pos_h.at[2, :3].set(
            data.site_xpos[self._feet_site_id[1]]
        )
        sup2navi_pos = (jp.linalg.inv(navi2world_pose) @ sup2world_pos_h.T).T[:, :3]

        foot2com_err = sup2navi_pos[1:] - sup2navi_pos[0]
        foot_center = foot2com_err[0, :2] + foot2com_err[1, :2]
        cost_support = jp.sum(jp.square(foot_center))
        cost_support *= stance_mask
        return cost_support

    def _cost_smoothness_action(self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array) -> jax.Array:
        smooth_0th = jp.square(act)
        smooth_1st = jp.square(act - last_act)
        smooth_2nd = jp.square(act - 2 * last_act + last_last_act)
        cost = jp.sum(smooth_0th + smooth_1st + smooth_2nd)
        return cost

    
@cat_ppo.registry.register("G1CatLoco", "command_to_reference_fn")
def command_to_reference(env_config: config_dict.ConfigDict, command: jax.Array):
    command_vel = command[1:]
    base_height = env_config.reward_config.base_height_target
    base_gvec = np.array([0.0, 0.0, 1.0])
    base_lin_vel = np.array([command_vel[0], command_vel[1], 0.0])
    base_ang_vel = np.array([0.0, 0.0, command_vel[2]])

    return {
        "base_height": base_height,
        "base_gvec": base_gvec,
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
    }
