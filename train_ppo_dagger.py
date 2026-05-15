import functools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import tyro
from absl import logging

import cat_ppo
from cat_ppo.constant import get_latest_ckpt
from cat_ppo.learning.policy.ppo import train as ppo
from train_ppo import (
    Args,
    _apply_args_to_config,
    _init_wandb,
    _log_checkpoint_path,
    _prepare_exp_name,
    _prepare_training_params,
    _progress,
    _report_training_time,
    _setup_paths,
    _validate_exp_name_format,
)


@dataclass
class DaggerArgs(Args):
    teacher_restore_names: list[str] = field(default_factory=list)


def _dagger_task_name(task: str) -> str:
    if task == "G1Cat":
        return "G1CatDagger"
    return task


def _checkpoint_config_path(ckpt_path: Path) -> Path:
    if ckpt_path.name.isdigit():
        return ckpt_path.parent / "config.json"
    if ckpt_path.name == "checkpoints":
        return ckpt_path / "config.json"
    return ckpt_path / "checkpoints" / "config.json"


def _prepare_dagger_config(policy_cfg, env_config, args: DaggerArgs):
    dagger_cfg = getattr(policy_cfg, "dagger_config", None)
    if dagger_cfg is None or not getattr(dagger_cfg, "enable", False):
        raise ValueError("train_ppo_dagger requires a task with enabled dagger_config")

    teacher_names = list(args.teacher_restore_names) or list(getattr(dagger_cfg, "teacher_restore_names", []))
    if not teacher_names:
        raise ValueError("Set --teacher_restore_names")
    teacher_checkpoint_paths = []
    for teacher_name in teacher_names:
        ckpt = get_latest_ckpt(teacher_name)
        if ckpt is None:
            raise ValueError(f"No checkpoint found for DAgger teacher run: {teacher_name}")
        teacher_checkpoint_paths.append(str(ckpt))

    scene_paths = []
    for ckpt in teacher_checkpoint_paths:
        config_path = _checkpoint_config_path(Path(ckpt))
        if not config_path.exists():
            raise ValueError(f"Missing teacher config.json for DAgger checkpoint: {ckpt}")
        with config_path.open("r", encoding="utf-8") as f:
            teacher_config = json.load(f)
        teacher_pf_config = teacher_config["env_config"]["pf_config"]
        teacher_dx = teacher_pf_config.get("dx", env_config.pf_config.dx)
        if float(teacher_dx) != float(env_config.pf_config.dx):
            raise ValueError(
                f"Teacher scene dx mismatch for {ckpt}: teacher dx={teacher_dx}, "
                f"student dx={env_config.pf_config.dx}"
            )
        scene_paths.append(teacher_pf_config["path"])

    env_config.pf_config.paths = scene_paths
    env_config.pf_config.path = scene_paths[0]
    env_config.pf_config.origin = [-0.5, -1.0, 0.0]
    dagger_cfg.teacher_restore_names = teacher_names
    dagger_cfg.teacher_checkpoint_paths = teacher_checkpoint_paths


def train(args: DaggerArgs):
    task_name = _dagger_task_name(args.task)
    env_class = cat_ppo.registry.get(task_name, "train_env_class")
    task_cfg = cat_ppo.registry.get(task_name, "config")
    env_cfg = task_cfg.env_config
    policy_cfg = task_cfg.policy_config

    base_args = {
        key: value
        for key, value in args.__dict__.items()
        if key != "teacher_restore_names"
    }
    train_args = Args(**{**base_args, "task": task_name})

    exp_name = _prepare_exp_name(task_name, train_args.generate_exp_name())
    debug_mode = "debug" in exp_name

    _validate_exp_name_format(exp_name, debug_mode)

    logdir, ckpt_path = _setup_paths(exp_name)
    _log_checkpoint_path(ckpt_path)

    _apply_args_to_config(train_args, policy_cfg, env_cfg, debug_mode)
    _prepare_dagger_config(policy_cfg, env_cfg, args)
    task_cfg.env_config = env_cfg
    policy_params = _prepare_training_params(policy_cfg, ckpt_path)

    if not debug_mode:
        _init_wandb(train_args, exp_name, env_class, task_cfg, ckpt_path)

    train_fn = functools.partial(ppo.train, **policy_params)
    times = [time.monotonic()]

    env = env_class(task_type=env_cfg.task_type, config=env_cfg)
    eval_env = env_class(task_type=env_cfg.task_type, config=env_cfg)

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=lambda s, m: _progress(s, m, times, policy_cfg.num_timesteps, debug_mode, exp_name),
        eval_env=eval_env,
        policy_params_fn=lambda *args: None,
    )

    _report_training_time(times)
    inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    logging.info(f"Run {exp_name} DAgger train done.")

    if train_args.convert_onnx:
        try:
            from cat_ppo.eval.brax2onnx import convert_jax2onnx, get_latest_ckpt as get_latest_saved_ckpt

            ckpt_dir = get_latest_saved_ckpt(ckpt_path)
            obs_size = {
                "privileged_state": (env_cfg.num_pri,),
                "state": (env_cfg.num_obs,),
            }
            act_size = env_cfg.num_act
            policy_obs_key = policy_cfg.network_factory.policy_obs_key
            convert_jax2onnx(
                ckpt_dir=ckpt_dir,
                output_path=f"{ckpt_dir}/policy.onnx",
                inference_fn=inference_fn,
                hidden_layer_sizes=policy_cfg.network_factory.policy_hidden_layer_sizes,
                obs_size=obs_size,
                action_size=act_size,
                policy_obs_key=policy_obs_key,
                jax_params=params,
                activation="swish",
            )
        except ImportError:
            logging.warning(
                "TensorFlow is not installed. Please install TensorFlow to use ONNX conversion."
            )


if __name__ == "__main__":
    train(tyro.cli(DaggerArgs))
