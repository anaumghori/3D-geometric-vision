import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import time
import os


def load_images(root):
    def load(name):
        path = os.path.join(root, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected file not found: {path}")
        return Image.open(path)

    ref_view = load("ref_view.png")
    secondary_view = load("secondary_view.png")
    disp_ref = load("disp_ref.png")

    return ref_view, secondary_view, disp_ref


def left_right_consistency_check(disp_lr, disp_rl, threshold=1.0):
    """Invalidate pixels where left-right and right-left disparity maps disagree.

    For each pixel (y, x) in disp_lr with disparity d, the corresponding pixel
    in disp_rl should be at column (x - d) and have approximately the same
    disparity. Pixels that fail this check are set to 0 (invalid)

    threshold: maximum allowed disparity difference in pixels
    """
    h, w = disp_lr.shape
    y_coords = np.tile(np.arange(h)[:, None], (1, w))
    x_coords = np.tile(np.arange(w)[None, :], (h, 1))
    x_right = np.round(x_coords - disp_lr).astype(np.int32)
    in_bounds = (x_right >= 0) & (x_right < w)
    x_right_clamped = np.clip(x_right, 0, w - 1)
    disp_rl_sampled = disp_rl[y_coords, x_right_clamped]
    consistent = in_bounds & (np.abs(disp_rl_sampled - disp_lr) <= threshold)
    result = disp_lr.copy()
    result[~consistent] = 0
    return result


def fill_disparity_holes(disparity):
    """
    For each invalid pixel on a scanline, fill with the smaller of the two nearest valid disparities
    on either side. Smaller disparity = farther away, which is the conservative background assumption
    for occluded regions. Pixels at the image border with no valid neighbor on one side use the
    available neighbor only.
    """
    filled = disparity.copy()
    h, w = filled.shape
    for y in range(h):
        row = filled[y]
        x = 0
        while x < w:
            if row[x] == 0:
                hole_start = x
                while x < w and row[x] == 0:
                    x += 1
                hole_end = x

                left_val = row[hole_start - 1] if hole_start > 0 else 0.0
                right_val = row[hole_end] if hole_end < w  else 0.0

                if left_val > 0 and right_val > 0:
                    fill_val = min(left_val, right_val)
                elif left_val > 0:
                    fill_val = left_val
                elif right_val > 0:
                    fill_val = right_val
                else:
                    fill_val = 0.0

                row[hole_start:hole_end] = fill_val
            else:
                x += 1

    return filled


def save_point_cloud(filename, disparity, colors, dmin, focal_length, baseline, min_disparity=10):
    """Generate 3D point cloud from disparity map and save to PLY format file.

    The direct approach used here:
        X = pixel column
        Y = -pixel row  (negated so Y increases upward)
        Z = disparity   (larger disparity = closer = larger Z)

    min_disparity: exclude pixels below this raw value. Near-zero disparities
    are typically SGM failures in textureless regions and add noise.
    """
    h, w = disparity.shape
    v_coords, u_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    mask = disparity >= min_disparity
    cx = w / 2.0
    cy = h / 2.0
    true_disp = disparity[mask].astype(np.float32) + dmin
    Z = (focal_length * baseline) / true_disp
    X = (u_coords[mask].astype(np.float32) - cx) * Z / focal_length
    Y = -(v_coords[mask].astype(np.float32) - cy) * Z / focal_length
    rgb = colors[mask]
    pts = np.column_stack([X, Y, Z, rgb])

    header = (
        f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(filename, "w") as f:
        f.write(header)
        np.savetxt(f, pts, fmt="%f %f %f %d %d %d")


def compute_metrics(pred, gt, valid_mask=None, threshold=3.0):
    """Compute disparity error metrics against ground truth.

    valid_mask: if provided, additionally excludes these pixels (e.g. pixels
    invalidated by LR consistency check). GT mask (gt > 0) is always applied.
    """
    mask = gt > 0
    if valid_mask is not None:
        mask = mask & valid_mask
    vp = pred[mask].astype(np.float32)
    vg = gt[mask].astype(np.float32)
    abs_err = np.abs(vp - vg)
    return {
        "EPE": round(float(np.mean(abs_err)), 2),
        "Bad Pixel %": round(float(np.sum(abs_err > threshold) / len(vg) * 100), 2),
        "RMSE": round(float(np.sqrt(np.mean((vp - vg) ** 2))), 2),
        "MAE": round(float(np.mean(abs_err)), 2),
    }


class SGM:
    def __init__(self, kernel_size, max_disparity, penalty1, penalty2, subpixel_interpolation):
        self.kernel_size = kernel_size
        self.kernel_half = kernel_size // 2
        self.max_disparity = max_disparity
        self.penalty1 = penalty1
        self.penalty2 = penalty2
        self.subpixel_interpolation = subpixel_interpolation

        if kernel_size * kernel_size > 64:
            raise ValueError("kernel_size > 8 exceeds 64-bit census capacity.")

    def _census_transform(self, img):
        """Census transform with uint64 (int32 overflows for kernel_size=7, 49 bits)."""
        h, w = img.shape
        kh = self.kernel_half
        n_bits = self.kernel_size * self.kernel_size
        bit_weights = (1 << np.arange(n_bits, dtype=np.uint64)[::-1])
        census = np.zeros((h, w), dtype=np.uint64)
        for y in range(kh, h - kh):
            for x in range(kh, w - kh):
                patch = img[y - kh:y + kh + 1, x - kh:x + kh + 1].flatten()
                bits = (patch > img[y, x]).astype(np.uint64)
                census[y, x] = bits.dot(bit_weights)
        return census

    def _compute_costs(self, left_census, right_census):
        """Hamming-distance cost volume. census_tmp reset each iteration."""
        h, w = left_census.shape
        kh = self.kernel_half
        cost_volume = np.zeros((h, w, self.max_disparity), dtype=np.uint32)
        for d in range(self.max_disparity):
            census_tmp = np.zeros((h, w), dtype=np.uint64)
            src_end = w - kh - d
            if src_end > kh:
                census_tmp[:, kh + d:w - kh] = right_census[:, kh:src_end]
            xor = np.bitwise_xor(left_census, census_tmp)
            cost_volume[:, :, d] = np.bitwise_count(xor).astype(np.uint32)
        return cost_volume

    def _build_penalty_matrix(self):
        D = self.max_disparity
        p2 = np.full((D, D), self.penalty2, dtype=np.int32)
        p1_band = np.full((D, D), self.penalty1 - self.penalty2, dtype=np.int32)
        p1_band = np.tril(np.triu(p1_band, k=-1), k=1)
        no_penalty = np.eye(D, dtype=np.int32) * -self.penalty1
        return p1_band + p2 + no_penalty

    def _get_path_cost(self, path_slice, penalties, n):
        cost_path = np.zeros((n, self.max_disparity), dtype=np.int32)
        cost_path[0] = path_slice[0]
        for i in range(1, n):
            prev = cost_path[i - 1]
            min_costs = (prev[:, None] + penalties).min(axis=0)
            cost_path[i] = path_slice[i] + min_costs - prev.min()
        return cost_path

    def _aggregate_costs(self, cost_volume):
        h, w, D = cost_volume.shape
        kh = self.kernel_half
        penalties = self._build_penalty_matrix()
        total = np.zeros((h, w, D), dtype=np.float32)

        for x in range(kh, w - kh):
            col = cost_volume[:, x, :].astype(np.int32)
            s = self._get_path_cost(col, penalties, h)
            n = np.flip(self._get_path_cost(np.flip(col, axis=0), penalties, h), axis=0)
            total[:, x, :] += s + n

        for y in range(kh, h - kh):
            row = cost_volume[y, :, :].astype(np.int32)
            e = self._get_path_cost(row, penalties, w)
            ww = np.flip(self._get_path_cost(np.flip(row, axis=0), penalties, w), axis=0)
            total[y, :, :] += e + ww

        return total

    def _apply_subpixel_offset(self, volume, disparity):
        h, w, D = volume.shape
        kh = self.kernel_half
        for i in range(kh, h - kh):
            for j in range(kh, w - kh):
                d = int(disparity[i, j])
                if 0 < d < D - 1:
                    denom = volume[i, j, d - 1] + volume[i, j, d + 1] - 2 * volume[i, j, d]
                    if denom != 0:
                        disparity[i, j] += (volume[i, j, d - 1] - volume[i, j, d + 1]) / (2 * denom)
        return disparity

    def compute(self, left, right):
        left_census = self._census_transform(left)
        right_census = self._census_transform(right)
        cost_volume = self._compute_costs(left_census, right_census)
        aggregated = self._aggregate_costs(cost_volume)
        disparity = np.argmin(aggregated, axis=2).astype(np.float32)
        if self.subpixel_interpolation:
            disparity = self._apply_subpixel_offset(aggregated, disparity)
        kh = self.kernel_half
        h, w = disparity.shape
        disparity[:kh, :] = 0
        disparity[h - kh:, :] = 0
        disparity[:, :kh] = 0
        disparity[:, w - kh:] = 0

        return disparity


if __name__ == "__main__":
    data_path = "data/dolls/"
    dataset_name = os.path.basename(os.path.normpath(data_path))
    max_disparity = 240
    kernel_size = 7
    penalty1 = 10
    penalty2 = 120
    subpixel_interpolation = True
    min_disparity_for_ply = 20
    lr_consistency_threshold = 1.0
    focal_length = 3740.0   # pixels
    baseline = 160.0    # mm

    ref_img, sec_img, gt_img = load_images(data_path)
    gt_arr = np.array(gt_img)
    print(f"  GT: dtype = {gt_arr.dtype}, range = [{gt_arr.min()}, {gt_arr.max()}], "
          f"non-zero = {np.sum(gt_arr > 0)}")

    dmin_path = os.path.join(data_path, "dmin.txt")
    with open(dmin_path, "r") as f:
        dmin = float(f.read().strip())

    ref_gray = np.array(ref_img.convert("L"))
    sec_gray = np.array(sec_img.convert("L"))
    matcher = SGM(kernel_size, max_disparity, penalty1, penalty2, subpixel_interpolation)
    start = time.time()
    disp_lr = matcher.compute(ref_gray, sec_gray)
    elapsed_lr = time.time() - start
    print(f"  LR disparity time = {elapsed_lr:.2f}s, range = [{disp_lr.min():.1f}, {disp_lr.max():.1f}]")

    start = time.time()
    disp_rl_flip = matcher.compute(np.fliplr(sec_gray), np.fliplr(ref_gray))
    disp_rl = np.fliplr(disp_rl_flip)
    elapsed_rl = time.time() - start
    print(f"  RL disparity time = {elapsed_rl:.2f}s, range = [{disp_rl.min():.1f}, {disp_rl.max():.1f}]")

    disparity_checked = left_right_consistency_check(disp_lr, disp_rl, threshold=lr_consistency_threshold)
    valid_mask = disparity_checked > 0
    valid_ratio = np.sum(valid_mask) / disparity_checked.size * 100
    print(f"  After LR check: valid pixels = {valid_ratio:.1f}%, "
          f"range = [{disparity_checked.min():.1f}, {disparity_checked.max():.1f}]")

    disparity_filled = fill_disparity_holes(disparity_checked)
    filled_ratio = np.sum(disparity_filled > 0) / disparity_filled.size * 100
    print(f"  After hole fill: valid pixels = {filled_ratio:.1f}%, "
          f"range = [{disparity_filled.min():.1f}, {disparity_filled.max():.1f}]")

    disparity_filtered = median_filter(disparity_filled, size=3).astype(np.float32)
    print(f"  Filtered range: [{disparity_filtered.min():.1f}, {disparity_filtered.max():.1f}]")

    m_raw = compute_metrics(disp_lr, gt_arr)
    m_check = compute_metrics(disparity_checked, gt_arr, valid_mask)
    m_filt = compute_metrics(disparity_filtered, gt_arr, valid_mask)
    print(f"\nMetrics vs GT:")
    print(f"  Raw LR only: {m_raw}")
    print(f"  After LR check (valid pixels only): {m_check}")
    print(f"  After filter   (valid pixels only): {m_filt}")

    os.makedirs("output", exist_ok=True)
    colors = np.array(ref_img)

    save_point_cloud(
        f"output/{dataset_name}_sgm.ply",
        disparity_filtered, colors,
        dmin, focal_length, baseline,
        min_disparity_for_ply
    )

    vis = disparity_filtered.copy()
    if vis.max() > 0:
        vis = 255.0 * vis / vis.max()
    plt.imsave(f"output/{dataset_name}_sgm.png", vis, cmap="jet")