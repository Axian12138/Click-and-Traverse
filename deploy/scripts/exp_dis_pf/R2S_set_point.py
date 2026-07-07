import numpy as np

def crop_rotated_box(occ, p1, p2, z_start=0,
                     size_x=128, size_y=128, size_z=35):
                    #  size_x=75, size_y=50, size_z=38):
    """
    occ: (H, W, D) 占据栅格
    p1, p2: 在 xy 平面上的两个点坐标 (2,), 单位: voxel index
    z_start: 在原栅格中 z 方向的起始 index
    返回: rotated crop, shape = (size_x, size_y, size_z)
    """

    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)

    # 中点 + 局部坐标轴
    c = p1
    d = p2 - p1
    norm = np.linalg.norm(d) + 1e-8
    ex = d / norm                        # 沿线段方向
    ey = np.array([-ex[1], ex[0]])       # 垂直方向

    # 局部格子坐标 (size_x * size_y)，以 c 为中心
    xs = np.arange(size_x, dtype=np.float32) - (size_x / 2.0 - 0.5)
    ys = np.arange(size_y, dtype=np.float32) - (size_y / 2.0 - 0.5)

    Xs, Ys = np.meshgrid(xs, ys, indexing='ij')  # (size_x, size_y)

    # 映射回原 xy
    dx = Xs * ex[0] + Ys * ey[0]
    dy = Xs * ex[1] + Ys * ey[1]

    x = c[0] + dx
    y = c[1] + dy

    # z 方向
    z = z_start + np.arange(size_z, dtype=np.float32)

    # broadcast 成 (size_x, size_y, size_z)
    x = x[..., None]
    y = y[..., None]
    z = z[None, None, :]

    # 最近邻索引 + 边界裁剪
    xi = np.clip(np.round(x).astype(int), 0, occ.shape[0] - 1)
    yi = np.clip(np.round(y).astype(int), 0, occ.shape[1] - 1)
    zi = np.clip(z.astype(int),          0, occ.shape[2] - 1)

    crop = occ[xi, yi, zi]  # (size_x, size_y, size_z)
    return crop
import matplotlib.pyplot as plt

voxel_size = 0.04      # m
segment_len_m = 2.0    # 2m
segment_len_vox = segment_len_m / voxel_size  # = 50

def show_topdown(occ):
    proj = occ.max(axis=2)
    return proj


def onclick(event):
    if not event.inaxes:
        return

    x, y = event.xdata, event.ydata  # float, 在图像坐标里
    print(f"Clicked: x={x:.2f}, y={y:.2f}")

    # 第一次点击：记录 p1
    if state["p1"] is None:
        state["p1"] = np.array([x, y], dtype=np.float32)
        ax.scatter([x], [y], c='r')
        ax.set_title("Now click a second point to give direction")
        fig.canvas.draw()
        return

    # 第二次点击：给方向
    p1 = state["p1"]
    p2_raw = np.array([x, y], dtype=np.float32)
    v = p2_raw - p1
    norm = np.linalg.norm(v)

    if norm < 1e-6:
        print("Second point too close to first, ignoring.")
        return

    dir_unit = v / norm
    p2 = p1 + dir_unit * segment_len_vox  # 真正长度为2m的端点

    # 画出实际使用的线段
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-')
    ax.scatter([p2[0]], [p2[1]], c='g')
    fig.canvas.draw()

    # 做 crop（z_start 随便先给个0，之后你可以做滑条）
    crop = crop_rotated_box(occ, p2, p1)
    np.save(obj_path, crop)
    print("Crop shape:", crop.shape)  # (75, 50, 38)

    # 展示一下 crop
    fig2, ax2 = plt.subplots(1, 3, figsize=(12, 4))
    midz = crop.shape[2] // 2
    ax2[0].imshow(crop[:, :, midz], origin='lower')
    ax2[0].set_title("Crop middle z slice")

    ax2[1].imshow(crop.max(axis=2), origin='lower')
    ax2[1].set_title("Crop max proj (z)")

    ax2[2].imshow(crop.mean(axis=2), origin='lower')
    ax2[2].set_title("Crop mean proj (z)")

    plt.show()

    # 重置，方便再选一段
    state["p1"] = None
    ax.set_title("Click first endpoint, then a direction point")
    fig.canvas.draw()


if __name__ == '__main__':
    occ_path = 'data/assets/R2SObs/uhln/voxel_1777284034.5574436.npy' # obs_pf_save_path in octomap_bridge.py
    obj_path = 'data/assets/R2SObs/uhln/voxel.npy'
    occ=np.load(occ_path)
    proj = show_topdown(occ)

    fig, ax = plt.subplots()
    im = ax.imshow(proj.T, origin='lower', cmap='gray')
    ax.set_title("Click first endpoint, then a direction point")

    state = {"p1": None}   # 保存第一次点击
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
