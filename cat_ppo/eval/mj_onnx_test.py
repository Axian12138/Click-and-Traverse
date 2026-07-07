import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=true"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import onnxruntime as rt
import tyro

import cat_ppo


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_ROOT = REPO_ROOT / "data/assets/RandObsEval"


@dataclass
class Args:
    task: str
    exp_name: str | None = None
    seed: int = 42
    onnx_path: str | None = None
    pri: bool = False
    obs_path: str = "data/assets/TypiObs/empty"
    obs_paths: list[str] = field(default_factory=list)
    obs_root: str | None = None
    manifest: str | None = None
    episodes: int = 50
    max_steps: int = 300
    success_x: float = 1.6
    recreate_env_each_episode: bool = True
    output_json: str | None = None
    generate_eval_set: bool = False
    generate_only: bool = False
    eval_root: str = str(DEFAULT_EVAL_ROOT)
    eval_difficulties: list[float] = field(default_factory=lambda: [0.8])
    eval_seeds: list[int] = field(default_factory=lambda: [101, 102, 103, 104, 105])
    eval_ground_counts: list[int] = field(default_factory=lambda: [2])
    eval_lateral_counts: list[int] = field(default_factory=lambda: [3])
    eval_overhead_counts: list[int] = field(default_factory=lambda: [2])
    grid_config_path: str | None = None


@dataclass
class EpisodeResult:
    success: bool
    reason: str
    steps: int
    final_x: float
    progress: float


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _scene_name(path: str | Path) -> str:
    return Path(path).name.rstrip("/")


def _is_scene_dir(path: Path) -> bool:
    required = ("sdf.npy", "bf.npy", "gf.npy")
    return path.is_dir() and all((path / name).exists() for name in required)


def _load_manifest(path: str | Path) -> list[str]:
    manifest_path = _resolve_path(path)
    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            data = data.get("scenes", [])
        return [str(item["path"] if isinstance(item, dict) else item) for item in data]

    scenes: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        scenes.append(line)
    return scenes


def _collect_scene_paths(args: Args) -> list[str]:
    scenes: list[str] = []
    if args.manifest:
        scenes.extend(_load_manifest(args.manifest))
    if args.obs_root:
        root = _resolve_path(args.obs_root)
        scenes.extend(str(path.relative_to(REPO_ROOT)) for path in sorted(root.iterdir()) if _is_scene_dir(path))
    scenes.extend(args.obs_paths)
    if not scenes:
        scenes.append(args.obs_path)
    return scenes


def _onnx_path(args: Args) -> Path:
    if args.onnx_path is not None:
        return _resolve_path(args.onnx_path)
    if args.exp_name is None:
        raise ValueError("Set either --onnx_path or --exp_name.")
    ckpt_path = cat_ppo.get_latest_ckpt(args.exp_name)
    if ckpt_path is None:
        raise ValueError(f"No checkpoint found for exp_name={args.exp_name!r}")
    return Path(ckpt_path) / "policy.onnx"


def _make_policy(args: Args) -> rt.InferenceSession:
    return rt.InferenceSession(str(_onnx_path(args)), providers=["CPUExecutionProvider"])


def _make_env(args: Args, obs_path: str):
    env_class = cat_ppo.registry.get(args.task, "play_env_class")
    task_cfg = cat_ppo.registry.get(args.task, "config")
    env_cfg = task_cfg.env_config
    env_cfg.pf_config.path = obs_path
    try:
        env = env_class(task_type=env_cfg.task_type, config=env_cfg, headless=True)
    except TypeError:
        env = env_class(task_type=env_cfg.task_type, config=env_cfg)
    env.pri = args.pri
    return env


def _run_episode(
    env,
    policy: rt.InferenceSession,
    output_names: list[str],
    max_steps: int,
    success_x: float,
) -> EpisodeResult:
    state = env.reset()
    goal_x = max(success_x, float(getattr(env, "current_goal_global", np.array([success_x]))[0]) - 0.2)
    for step in range(1, max_steps + 1):
        obs = state.obs["state"].reshape(1, -1).astype(np.float32)
        action = policy.run(output_names, {"obs": obs})[0][0]
        state = env.step(state, action)
        final_x = float(env.mj_data.qpos[0])
        progress = float(np.clip(final_x / max(goal_x, 1e-6), 0.0, 1.0))
        if bool(getattr(env, "done", False)) or final_x >= success_x:
            return EpisodeResult(True, "success", step, final_x, 1.0)
        if not bool(state.info.get("succ", True)):
            return EpisodeResult(False, "failure", step, final_x, progress)
    final_x = float(env.mj_data.qpos[0])
    progress = float(np.clip(final_x / max(goal_x, 1e-6), 0.0, 1.0))
    return EpisodeResult(False, "timeout", max_steps, final_x, progress)


def evaluate_scene(args: Args, policy: rt.InferenceSession, obs_path: str) -> dict[str, Any]:
    output_names = ["continuous_actions"]
    env = None if args.recreate_env_each_episode else _make_env(args, obs_path)
    episodes = []
    for _ in range(args.episodes):
        episode_env = _make_env(args, obs_path) if args.recreate_env_each_episode else env
        episodes.append(_run_episode(episode_env, policy, output_names, args.max_steps, args.success_x))
    success = np.array([episode.success for episode in episodes], dtype=np.float32)
    timeout = np.array([episode.reason == "timeout" for episode in episodes], dtype=np.float32)
    failure = np.array([episode.reason == "failure" for episode in episodes], dtype=np.float32)
    steps = np.array([episode.steps for episode in episodes], dtype=np.float32)
    progress = np.array([episode.progress for episode in episodes], dtype=np.float32)
    return {
        "scene": _scene_name(obs_path),
        "obs_path": obs_path,
        "episodes": args.episodes,
        "success_rate": float(success.mean()),
        "failure_rate": float(failure.mean()),
        "timeout_rate": float(timeout.mean()),
        "mean_steps": float(steps.mean()),
        "mean_progress": float(progress.mean()),
    }


def _write_manifest(eval_root: Path, scene_paths: list[Path]) -> Path:
    manifest_path = eval_root / "manifest.json"
    payload = {
        "root": str(eval_root.relative_to(REPO_ROOT)),
        "scenes": [{"name": path.name, "path": str(path.relative_to(REPO_ROOT))} for path in scene_paths],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _generate_one_eval_scene(
    out_root: Path,
    difficulty: float,
    seed: int,
    n_rect_f: int,
    n_rect_l: int,
    n_rect_c: int,
    grid_config_path: str | None,
) -> Path | None:
    proc_dir = REPO_ROOT / "procedural_obstacle_generation"
    sys.path.insert(0, str(proc_dir))
    from main import better_mesh
    from pf_modular import PFConfig, grad3, make_guidance_field_progressive, make_sdf
    from random_obstacle import Cfg as ObsCfg
    from random_obstacle import generate_and_save, make_axes

    prefix = f"D{int(difficulty * 10)}G{n_rect_f:01d}L{n_rect_l:01d}O{n_rect_c:01d}S{seed}"
    scene_dir = out_root / prefix
    scene_dir.mkdir(parents=True, exist_ok=True)
    obs_cfg = ObsCfg(
        grid_config_path=grid_config_path,
        difficulty=difficulty,
        seed=seed,
        n_rect_L=n_rect_l,
        n_rect_R=n_rect_l,
        n_rect_F=n_rect_f,
        n_rect_C=n_rect_c,
    )
    obs_mask, xv, yv, zv = generate_and_save(obs_cfg, prefix=str(scene_dir / "tmp"), save=False)
    if not obs_mask.any():
        return None

    pf_cfg = PFConfig(grid_config_path=grid_config_path)
    mesh = better_mesh((pf_cfg.voxel, pf_cfg.voxel, pf_cfg.voxel), obs_mask)
    mesh.export(scene_dir / "obs.obj")
    sdf = make_sdf(obs_mask, pf_cfg.voxel)
    bf = grad3(sdf, pf_cfg.voxel)
    x, y, z = np.meshgrid(xv, yv, zv, indexing="ij")
    _, gf = make_guidance_field_progressive(pf_cfg, (x, y, z), obs_mask, pf_cfg.goal_w, bf, sdf)
    np.save(scene_dir / "sdf.npy", sdf)
    np.save(scene_dir / "bf.npy", bf)
    np.save(scene_dir / "gf.npy", gf)
    np.save(scene_dir / "obs.npy", obs_mask.astype(np.uint8))
    return scene_dir


def generate_eval_set(args: Args) -> Path:
    eval_root = _resolve_path(args.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    scene_paths: list[Path] = []
    for difficulty in args.eval_difficulties:
        for seed in args.eval_seeds:
            for n_rect_f in args.eval_ground_counts:
                for n_rect_l in args.eval_lateral_counts:
                    for n_rect_c in args.eval_overhead_counts:
                        scene_dir = _generate_one_eval_scene(
                            eval_root,
                            difficulty,
                            seed,
                            n_rect_f,
                            n_rect_l,
                            n_rect_c,
                            args.grid_config_path,
                        )
                        if scene_dir is not None:
                            scene_paths.append(scene_dir)
    manifest_path = _write_manifest(eval_root, scene_paths)
    print(f"Generated {len(scene_paths)} eval scenes under {eval_root}")
    print(f"Manifest: {manifest_path}")
    return manifest_path


def play(args: Args):
    if args.generate_eval_set:
        manifest_path = generate_eval_set(args)
        if args.generate_only:
            return
        if not args.manifest and not args.obs_root and not args.obs_paths:
            args.manifest = str(manifest_path)

    policy = _make_policy(args)
    scene_paths = _collect_scene_paths(args)
    results = []
    for index, obs_path in enumerate(scene_paths, start=1):
        print(f"[{index}/{len(scene_paths)}] Evaluating {obs_path}")
        result = evaluate_scene(args, policy, obs_path)
        results.append(result)
        print(
            f"  success={result['success_rate']:.3f} "
            f"failure={result['failure_rate']:.3f} "
            f"timeout={result['timeout_rate']:.3f} "
            f"progress={result['mean_progress']:.3f}"
        )

    success_rates = np.array([result["success_rate"] for result in results], dtype=np.float32)
    summary = {
        "args": asdict(args),
        "num_scenes": len(results),
        "mean_success_rate": float(success_rates.mean()) if len(success_rates) else 0.0,
        "min_success_rate": float(success_rates.min()) if len(success_rates) else 0.0,
        "max_success_rate": float(success_rates.max()) if len(success_rates) else 0.0,
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    if args.output_json:
        output_path = _resolve_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    play(tyro.cli(Args))
