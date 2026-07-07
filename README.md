<div align="center">  
  <img src="assets/icon.png" width="150" style="vertical-align: middle;">
  <h1 align="center"> Click and Traverse </h1>
Collision-Free Humanoid Traversal in Cluttered Indoor Scenes
  <h3 align="center"> Tsinghua · GALBOT </h3>

[中文](README_zh.md) | [English](README.md)

📃[Paper](https://arxiv.org/abs/2601.16035) | 🏠[Website](https://axian12138.github.io/CAT/) | 📽[Video](https://www.youtube.com/watch?v=blek__Qf0Vc)
  </div>

## News

- 2026/07/05: We release **generalist v1**. This is the first generalist checkpoint for community testing; we will keep improving v2 based on feedback.
- 2026/05/20: We release the **specialist-to-generalist policy distillation code**.
- 2026/03/07: We release the **real-world deployment code** of CAT! Please refer to deploy/Click-and-Traverse-SLAM for details.
- 2026/01/08: We release the official implementation of CAT!

---

The project addresses the problem of enabling humanoid robots to safely traverse **cluttered indoor scenes**, which we define as environments that simultaneously exhibit:

- **Full-spatial constraints**: obstacles jointly present at the *ground*, *lateral*, and *overhead* levels, restricting the humanoid’s motion in all spatial dimensions.
- **Intricate geometries**: obstacles with complex, irregular shapes that go beyond simple primitives such as rectangular blocks or regular polyhedra.

<p align="center">
  <img src="assets/teaser.png" width="40%">
  <img src="assets/comparison.png" width="50%">
</p>

In this repository, we present:

- **Humanoid Potential Field (HumanoidPF)**: a structured representation encoding spatial relationships between the humanoid body and surrounding obstacles;
- **Hybrid scene generation**: realistic 3D indoor scene crops combined with procedurally synthesized obstacles;
- **Reinforcement learning for specialist and generalist policies**, respectively trained on specific scenes and distilled to a generalist policy.

<p align="center">
  <img src="assets/pipeline.png" width="95%">
</p>

## Table of Contents

- [Project Status](#project-status)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Hybrid Obstacle Generation &amp; HumanoidPF](#hybrid-obstacle-generation--humanoidpf)
- [Traversal Skill Learning](#traversal-skill-learning)
- [Related Projects](#related-projects)
- [Citation](#citation)
- [License](#license)
- [Acknowledgement](#acknowledgement)
- [Contact Us](#contact-us)

---

## Project Status

- [X] 🧩 Procedural obstacle generation and HumanoidPF construction
- [X] 🧩 Specialist policy training code
- [X] 🗂️ Pre-trained specialist models and scene data
- [X] 🚀 Real-world deployment code
- [X] 🧩 Real-to-sim contruction for sim2sim test and real-scene finetuning
- [X] 🧩 Specialist-to-generalist policy distillation code
- [X] 🗂️ Pre-trained generalist v1 model
- [ ] 🗂️ Expanded scene datasets

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/GalaxyGeneralRobotics/Click-and-Traverse.git
cd Click-and-Traverse
```

### 2. Environment setup

CUDA 12.5 is recommended.

```bash
export PATH=/usr/local/cuda-12.5/bin:$PATH  # adjust if needed
uv sync -i https://pypi.org/simple
```

### 3. Configuration

Create and customize the `.env` file in the repository root. This file defines runtime configurations such as:

- working directory paths
- logging (e.g., WandB account)
- experiment identifiers

### 4. Initialize MuJoCo assets

```bash
source .venv/bin/activate
source .env
python -m cat_ppo.utils.mj_playground_init
```

---

## Repository Structure

Pre-trained checkpoints and scene assets can be downloaded from:

- **Google Drive**: 
  - https://drive.google.com/drive/folders/1q57nJJ6uC26RmmCuxYjv6q1zE1gnVFvr

- **Tsinghua Cloud (domestic Chinese platform)**:
  - https://cloud.tsinghua.edu.cn/d/5a6b3c27259d4ae5b1dd/

- **Huggingface**: 
  - Models (logs): https://huggingface.co/Axian12138/Click-and-Traverse
  - Datasets (assets): https://huggingface.co/datasets/Axian12138/Click-and-Traverse

Place downloaded data under the `data/` directory.

```
Click-and-Traverse/
├── LICENSE
├── README.md
├── pyproject.toml
├── train_batch.py
├── train_ppo.py
├── .env
├── cat_ppo/                        # Core RL framework
│   ├── envs/
│   ├── learning/
│   ├── eval/
│   └── utils/
├── data/                           # Assets, logs (checkpoints)
│   ├── assets/
│   |   ├── mujoco_menagerie/       # after mj_playground_init
│   |   ├── RandObs/                # random obstacles, downloaded or generated
│   |   ├── RandObsEval/            # optional held-out random obstacles, generated locally
│   |   ├── TypiObs/                # typical obstacles
│   |   └── unitree_g1/             # humanoid assets
│   └── logs/
|       └── G1_mj_axis/             # downloaded checkpoints
├── eval_scene_specs/               # lightweight evaluation scene generation specs
├── deploy/                         # Real-world deployment
│   ├── gx_loco_deploy/             # deploy helpers
│   ├── scripts/
|   |   └── exp_dis_pf/   
│   └── Click-and-Traverse-SLAM/ 
└── procedural_obstacle_generation/ # Obstacle generation
    ├── main.py
    ├── pf_modular.py               # HumanoidPF construction
    ├── random_obstacle.py
    ├── typical_obstacle.py
    └── utils.py
```

---

## Hybrid Obstacle Generation & HumanoidPF

Two categories of obstacle scenes are supported:

- **Typical obstacles**: manually designed, semantically meaningful scenes
- **Random obstacles**: procedurally generated scenes with controllable difficulty

HumanoidPF representations are generated synchronously for all scenes.

Outputs are saved to:

- `data/assets/TypiObs/`
- `data/assets/RandObs/`

### Generate Typical Obstacles

```bash
export PATH=/usr/local/cuda-12.5/bin:$PATH
source .env
source .venv/bin/activate
cd procedural_obstacle_generation
```

Edit `main.py` and call:

```python
generate_typical_obstacle(obs_name)
```

Parameters:

- `obs_name`: the obstacle configuration (see comments in `main.py`).

### Generate Random Obstacles

Call in `main.py`:

```python
generate_random_obstacle(difficulty, seed, dL, dG, dO)
```

Parameters:

- `difficulty`: global difficulty level
- `seed`: random seed
- `dL`: lateral obstacle difficulty
- `dG`: ground obstacle difficulty
- `dO`: overhead obstacle difficulty

---

## Traversal Skill Learning

### Generalist v1 Release

We provide **generalist v1** as the first broadly usable policy distilled from specialist teachers. This version is intended for testing and feedback collection. We expect to release improved v2 checkpoints after collecting more failure cases and evaluation results from the community.

The current generalist v1 training split is the random-obstacle training set:

- `data/assets/RandObs/`
- 42 generated scenes in the current release.
- The exact scene list and training configuration can be found in the released checkpoint config, e.g. `data/logs/G1_mj_v2_axis/07020635_G1CatDagger_dagger_v8allD8RandObsX4xG2p0xL1p0xO1p0xT0p0/checkpoints/config.json`.


### Training Specialist

```bash
export PATH=/usr/local/cuda-12.5/bin:$PATH
source .env
source .venv/bin/activate
python train_batch.py
```

If you want to train a specific experiment, you can run:

```bash
python -m train_ppo --task {task} --restore_name {restore_name} --exp_name {exp_name}  --ground {ground} --lateral {lateral} --overhead {overhead} --term_collision_threshold {term_collision_threshold} --obs_path {obs_path}
```

Supported tasks:

- `G1Cat`: default task (can be directly used for sim-to-real deployment)
- `G1CatPri`: privileged task (privileged observation is more informative for distilling generalist policies)

Refer to `train_batch.py` for args details.

### Training Generalist

```bash
export PATH=/usr/local/cuda-12.5/bin:$PATH
source .env
source .venv/bin/activate
python train_dagger_batch.py
```

If you want to train a specific experiment, you can run:

```bash
python -m train_ppo_dagger --task {task} --restore_name {restore_name} --exp_name {exp_name}  --ground {ground} --lateral {lateral} --overhead {overhead} --term_collision_threshold {term_collision_threshold} --dagger_timesteps {dagger_timesteps} --obs_path {obs_path} --teacher_restore_names {teacher_restore_names}
```

Supported tasks:

- `G1Cat`: generalist student policy trained by specialist-to-generalist DAgger distillation

The teacher policies are specified by `--teacher_restore_names`. Refer to `train_dagger_batch.py` for args details.

### brax2onnx

`train_batch.py` will automatically convert checkpoints to ONNX format. But if you customize the policy architecture, you may need to convert checkpoints to ONNX manually:

```bash
python -m cat_ppo.eval.brax2onnx \
  --task G1CatPri \
  --exp_name 05200601_G1CatPri_V0_xT0p0xforward
```

### Evaluation

For interactive single-scene playback without privileged observation, run:

```bash
python -m cat_ppo.eval.mj_onnx_play --task G1Cat --exp_name 07020635_G1CatDagger_dagger_v8allD8RandObsX4xG2p0xL1p0xO1p0xT0p0 --obs_path data/assets/RandObsEval/D9G2L1O2S110
```

To evaluate the model with privileged observation, run:

```bash
python -m cat_ppo.eval.mj_onnx_play --task G1CatPri --pri --exp_name G1CatPri_side1 --obs_path data/assets/TypiObs/side1
```

For batch success-rate evaluation without visualization, use `mj_onnx_test`:

```bash
python -m cat_ppo.eval.mj_onnx_test \
  --task G1Cat \
  --exp-name 07020635_G1CatDagger_dagger_v8allD8RandObsX4xG2p0xL1p0xO1p0xT0p0 \
  --obs-root data/assets/RandObs \
  --episodes 1 \
  --output-json output_logs/mj_onnx_eval_randobs_train_freshenv.json
```

To evaluate a held-out test set:

```bash
python -m cat_ppo.eval.mj_onnx_test \
  --task G1Cat \
  --exp-name 07020635_G1CatDagger_dagger_v8allD8RandObsX4xG2p0xL1p0xO1p0xT0p0 \
  --manifest data/assets/RandObsEval/manifest.json \
  --episodes 1 \
  --output-json output_logs/mj_onnx_eval_randobs_eval_freshenv.json
```

If `data/assets/RandObsEval/` is not available, generate it locally from the lightweight spec in `eval_scene_specs/randobs_eval_v1.yaml`. Scene names are determined by the generation parameters, so users can recreate the same split on their own machines:

```bash
python -m cat_ppo.eval.mj_onnx_test \
  --task G1Cat \
  --generate-eval-set \
  --generate-only \
  --eval-root data/assets/RandObsEval \
  --eval-difficulties 0.6 0.7 0.8 0.9 \
  --eval-seeds 101 102 103 104 105 106 107 108 109 110 \
  --eval-ground-counts 1 2 3 \
  --eval-lateral-counts 1 3 \
  --eval-overhead-counts 1 2
```

The command writes generated assets to `data/assets/RandObsEval/` and creates `data/assets/RandObsEval/manifest.json`, which can then be passed to the evaluation command above.


---

## Related Projects

- [R2S2: Whole-body-control with various real-world-ready motor skills.](https://github.com/GalaxyGeneralRobotics/OpenWBT) & [code](https://github.com/GalaxyGeneralRobotics/OpenWBT)
- [Any2Track: Foundational motion tracking to track any motions under any disturbances.](https://zzk273.github.io/Any2Track/) & [code](https://github.com/GalaxyGeneralRobotics/OpenTrack)

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{xue2026collisionfreehumanoidtraversalcluttered,
  title        = {Collision-Free Humanoid Traversal in Cluttered Indoor Scenes},
  author       = {Xue, Han and Liang, Sikai and Zhang, Zhikai and Zeng, Zicheng and Liu, Yun and Lian, Yunrui and Wang, Jilong and Liu, Qingtao and Shi, Xuesong and Li, Yi},
  year         = {2026},
  eprint       = {2601.16035},
  archivePrefix= {arXiv},
  primaryClass = {cs.RO},
  url          = {https://arxiv.org/abs/2601.16035}
}
```

---

## License

This project is released under the terms of the LICENSE file included in this repository.

---

## Acknowledgement

We thank the MuJoCo Playground for providing a convenient simulation framework.

---

# Contact Us

If you'd like to discuss anything, feel free to send an email to xue-h21@mails.tsinghua.edu.cn or add WeChat: xh15158435129.

Contributions are welcome. Please open an issue to discuss major changes or submit a pull request directly.
