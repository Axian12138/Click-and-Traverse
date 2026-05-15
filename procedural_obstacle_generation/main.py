from random_obstacle import make_axes, generate_and_save, extract_surface_voxels, Cfg as ObsCfg, get_elevation
from typical_obstacle import build_obstacles
import os
import shutil
from pf_modular import make_sdf, make_guidance_field_progressive, grad3, PFConfig, visualize_all
import numpy as np
from utills import marching_cubes_mesh, occupancy_to_points, preview_matplotlib, combine_meshes
import itertools
from grid_config import expected_shape_from_config


def generate_random_obstacle(difficulty, seed, n_rect_R, n_rect_F, n_rect_C, grid_config_path=None):
    n_rect_L=n_rect_R
    prefix = f"D{int(difficulty*10)}G{n_rect_F:01d}L{n_rect_R:01d}O{n_rect_C:01d}S{seed}/"

    print(prefix)
    save = False
    if save:
        os.makedirs(prefix, exist_ok=True)
    obs_cfg = ObsCfg(
        grid_config_path=grid_config_path,
        difficulty=difficulty,
        seed=seed,
        n_rect_L=n_rect_L,
        n_rect_R=n_rect_R,
        n_rect_F=n_rect_F,
        n_rect_C=n_rect_C,
    )
    obs_mask, xv, yv, zv = generate_and_save(obs_cfg, prefix=prefix, save=save)
    if obs_mask.any() == False:
        shutil.rmtree(prefix)
        return

    os.makedirs(f"../data/assets/RandObs/{prefix}", exist_ok=True)
    # pts = occupancy_to_points(obs_mask, voxel_size=obs_cfg.voxel)
    # preview_matplotlib(pts)
    spacing = (obs_cfg.voxel, obs_cfg.voxel, obs_cfg.voxel)
    mesh = better_mesh(spacing, obs_mask)
    mesh.export(f"../data/assets/RandObs/{prefix}obs.obj")

    cfg = PFConfig(grid_config_path=grid_config_path)
    
    # SDF and gradient
    sdf = make_sdf(obs_mask, cfg.voxel)
    bf  = grad3(sdf, cfg.voxel)

    # HumanoidPF and gradient
    X, Y, Z = np.meshgrid(xv, yv, zv, indexing='ij')
    T, gf = make_guidance_field_progressive(cfg, (X, Y, Z), obs_mask, cfg.goal_w, bf, sdf)

    np.save(f"../data/assets/RandObs/{prefix}sdf.npy", sdf)
    np.save(f"../data/assets/RandObs/{prefix}bf.npy",  bf)
    np.save(f"../data/assets/RandObs/{prefix}gf.npy",  gf)
    np.save(f"../data/assets/RandObs/{prefix}obs.npy", obs_mask.astype(np.uint8))
    # sur = extract_surface_voxels(obs_mask)
    # np.save(f"../data/assets/RandObs/{prefix}sur.npy", sur.astype(np.uint8))
    # pts = occupancy_to_points(sur, voxel_size=obstacle_cfg.voxel)
    # preview_matplotlib(pts)

    os.makedirs(f"fig/", exist_ok=True)
    visualize_all(xv, yv, zv, sdf, T, gf, obs_mask, cfg.start_w, cfg.goal_w, title_prefix=f'fig/{prefix[:-1]}')

def generate_typical_obstacle(scene_type, grid_config_path=None):
    prefix = f"{scene_type}/"
    cfg = PFConfig(grid_config_path=grid_config_path)
    
    xv, yv, zv = make_axes(cfg)
    X, Y, Z = np.meshgrid(xv, yv, zv, indexing='ij')
    obs_mask = build_obstacles(scene_type, (X, Y, Z))

    os.makedirs(f"../data/assets/TypiObs/{prefix}", exist_ok=True)
    # pts = occupancy_to_points(obs_mask, voxel_size=cfg.voxel)
    # preview_matplotlib(pts)
    spacing = (cfg.voxel, cfg.voxel, cfg.voxel)
    mesh = better_mesh(spacing, obs_mask)
    mesh.export(f"../data/assets/TypiObs/{prefix}obs.obj")

    sdf = make_sdf(obs_mask, cfg.voxel)
    bf  = grad3(sdf, cfg.voxel)
    T, gf = make_guidance_field_progressive(cfg, (X, Y, Z), obs_mask, cfg.goal_w, bf, sdf)

    np.save(f"../data/assets/TypiObs/{prefix}sdf.npy", sdf)
    np.save(f"../data/assets/TypiObs/{prefix}bf.npy",  bf)
    np.save(f"../data/assets/TypiObs/{prefix}gf.npy",  gf)
    np.save(f"../data/assets/TypiObs/{prefix}obs.npy", obs_mask.astype(np.uint8))
    # sur = extract_surface_voxels(obs_mask)
    # np.save(f"../data/assets/TypiObs/{prefix}sur.npy", sur.astype(np.uint8))
    # ground_idx, ceil_idx = get_elevation(obs_mask)
    # np.save(f"../data/assets/TypiObs/{prefix}ground.npy", ground_idx)
    # np.save(f"../data/assets/TypiObs/{prefix}ceil.npy", ceil_idx)
    # pts = occupancy_to_points(sur, voxel_size=obs_cfg.voxel)
    # preview_matplotlib(pts)

    # visualize_all(xv, yv, zv, sdf, T, gf, obs_mask, cfg.start_w, cfg.goal_w)

def generate_pf(scene_type, pc_path, grid_config_path=None):
    prefix = f"{scene_type}/"
    cfg = PFConfig(grid_config_path=grid_config_path)
    expected_shape = expected_shape_from_config(grid_config_path)

    
    xv, yv, zv = make_axes(cfg)
    X, Y, Z = np.meshgrid(xv, yv, zv, indexing='ij')
    # obs_mask = torch.load(pc_path).cpu().numpy()
    obs_mask = np.load(pc_path, allow_pickle=True) 
    if obs_mask.dtype != np.bool_:
        obs_mask = obs_mask.astype(np.uint8) > 0
    if tuple(obs_mask.shape) != expected_shape:
        raise ValueError(
            f"obs_mask shape {obs_mask.shape} does not match PF grid shape {expected_shape}. "
            "Pass the matching voxel .meta.yaml or config path to generate_pf(..., grid_config_path=...)."
        )

    # some pre-processing for real-to-sim occupancy
    z_bar_thresh = 6
    mask_bar = obs_mask[:, :, :z_bar_thresh] == 1
    filled = np.cumsum(mask_bar[:, :, ::-1], axis=2)[:, :, ::-1] > 0
    obs_mask[:, :, :z_bar_thresh] = filled.astype(np.uint8)
    # obs_mask[:,0] = 1
    # obs_mask[:,-1] = 1

    os.makedirs(f"../data/assets/R2SObs/{prefix}", exist_ok=True)
    pts = occupancy_to_points(obs_mask, voxel_size=cfg.voxel)
    preview_matplotlib(pts)
    spacing = (cfg.voxel, cfg.voxel, cfg.voxel)
    mesh = better_mesh(spacing, obs_mask)
    mesh.export(f"../data/assets/R2SObs/{prefix}obs.obj")

    sdf = make_sdf(obs_mask, cfg.voxel)
    bf  = grad3(sdf, cfg.voxel)
    # Eikonal 
    T, gf = make_guidance_field_progressive(cfg, (X, Y, Z), obs_mask, cfg.goal_w, bf, sdf)

    # 保存
    np.save(f"../data/assets/R2SObs/{prefix}sdf.npy", sdf)
    np.save(f"../data/assets/R2SObs/{prefix}bf.npy",  bf)
    np.save(f"../data/assets/R2SObs/{prefix}gf.npy",  gf)
    np.save(f"../data/assets/R2SObs/{prefix}obs.npy", obs_mask.astype(np.uint8))
    # sur = extract_surface_voxels(obs_mask)
    # np.save(f"../data/assets/R2SObs/{prefix}sur.npy", sur.astype(np.uint8))
    # 可视化
    visualize_all(xv, yv, zv, sdf, T, gf, obs_mask, cfg.start_w, cfg.goal_w)


def better_mesh(spacing, obs_mask): # for mujoco visualization
    # obs_mask[:,0,:] = 0
    # obs_mask[:,-1,:] = 0
    # obs_mask[:,:,-1] = 0
    obs_mask_erosion = 1-obs_mask
    mesh = marching_cubes_mesh(obs_mask_erosion, spacing=spacing)
    return mesh

if __name__ == "__main__":
    # difficulties = [0.8]
    # d_Ls = [1,3]
    # d_Gs = [0, 1, 2]
    # d_Os = [0, 1, 2]
    # seeds = [1,2]
    # combos = itertools.product(difficulties, d_Ls, d_Gs, d_Os, seeds)
    # for difficulty, dL, dG, dO, seed in combos:
    #     rng = (dL + 0.5 * seed) * (dG + seed) * (dO + 1.5 * seed) + 1
    #     generate_random_obstacle(difficulty, int(rng), dL, dG, dO)
    # difficulties = [0.7]
    # d_Ls = [4]
    # d_Gs = [0, 1, 2]
    # d_Os = [0, 1, 2]
    # seeds = [1,2]
    # combos = itertools.product(difficulties, d_Ls, d_Gs, d_Os, seeds)
    # for difficulty, dL, dG, dO, seed in combos:
    #     rng = (dL + 0.5 * seed) * (dG + seed) * (dO + 1.5 * seed)
    #     generate_random_obstacle(difficulty, int(rng), dL, dG, dO)
    # generate_typical_obstacle('ceil1')
    # generate_typical_obstacle('bar0')
    # generate_typical_obstacle('bar1')
    # generate_typical_obstacle('bar2')
    # generate_typical_obstacle('bar3')
    # generate_typical_obstacle('Mceil0')
    # generate_typical_obstacle('Mceil1')
    # generate_typical_obstacle('Mbar0')
    # generate_typical_obstacle('Mbar1')
    # generate_typical_obstacle('ceilbar0')
    # generate_typical_obstacle('ceilbar1')
    # generate_typical_obstacle('chest')
    # generate_typical_obstacle('Nbar0')
    # generate_typical_obstacle('Nbar1')
    # generate_typical_obstacle('doubar')
    # generate_typical_obstacle('lowcorner')
    # generate_typical_obstacle('hole')
    # generate_random_obstacle(0.4, 13, 2, 1, 1)
    # generate_random_obstacle(0.8, 4, 1, 0, 1)
    generate_pf('oavHard0', '/home/galbot/workspace/oav/short/tasks/hard/scene_0000_diff80_seed202604116/export/scene.voxel.npy', '/home/galbot/workspace/oav/short/tasks/hard/scene_0000_diff80_seed202604116/export/scene.voxel.meta.yaml')
