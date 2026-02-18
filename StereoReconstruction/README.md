# Stereo Reconstruction with Semi-Global Matching

Stereo reconstruction is a computer vision technique that recovers 3D depth information from two images of the same scene taken from slightly different viewpoints, much like how human binocular vision works. By analyzing how objects shift position between the two images (a quantity called disparity), we can infer how far each point in the scene is from the camera.

The fundamental question stereo reconstruction answers is: **"Given two images from different viewpoints, how far away is each visible surface point?"** The answer to this question enables applications like autonomous navigation, robotic manipulation, and 3D scanning.

At its core, the problem is one of correspondence: for every pixel in the left image, find the matching pixel in the right image. The horizontal distance between these matched pixels is the disparity, and disparity is inversely proportional to depth - objects closer to the camera appear to shift more between views. While this sounds straightforward, the challenge lies in handling:
- **Textureless regions**: Smooth surfaces like walls provide no distinctive patterns to match
- **Occlusions**: Some surfaces visible in one image are hidden in the other
- **Reflective and transparent surfaces**: These violate the assumption that appearance is consistent across views
- **Repetitive patterns**: Repeating textures like brick walls create ambiguous matches


https://github.com/user-attachments/assets/8446efd1-49c2-4493-bbe9-46a8f61b01ed


### Semi-Global Matching (SGM)

Semi-Global Matching, introduced by Hirschmüller (2005), is one of the most widely used stereo matching algorithms in both academic and industrial applications. It strikes a balance between the speed of local methods (which compare small patches independently) and the accuracy of global methods (which optimize over the entire image but are computationally expensive).

SGM works by approximating a global 2D optimization with multiple 1D optimizations along different directions across the image. For each pixel and each possible disparity, it accumulates matching costs along several paths (typically 4 to 16 directions), applying smoothness penalties that discourage abrupt disparity changes while still allowing sharp depth discontinuities at object boundaries. The penalties are controlled by two parameters:
- **P1** (small penalty): Applied when disparity changes by exactly 1 pixel between neighbors, accommodating slanted or curved surfaces
- **P2** (large penalty): Applied for disparity changes greater than 1, preserving sharp depth edges while discouraging noise

The final disparity at each pixel is chosen as the one with the lowest total aggregated cost across all paths.


### Census Transform

The census transform is a non-parametric local descriptor used for computing the initial matching cost between pixels. Unlike simple intensity difference, the census transform encodes the spatial structure of a pixel's neighborhood into a binary string, making it highly robust to changes in lighting, exposure, and camera gain between the two views.

How it works:
1. **Local comparison**: For each pixel, compare its intensity against every neighbor within a window (e.g., 7×7)
2. **Binary encoding**: Encode each comparison as a single bit — 1 if the neighbor is brighter, 0 otherwise
3. **Bit string**: Concatenate all bits into a single binary descriptor representing the local intensity pattern
4. **Matching cost**: Compare descriptors from left and right images using the Hamming distance (count of differing bits)

The census transform captures the relative ordering of intensities rather than their absolute values, which means it remains stable even when the two cameras have different brightness levels or slightly different exposure settings.

### Left-Right Consistency Check

A critical post-processing step that validates disparity estimates by exploiting the symmetry of stereo matching. The idea is simple: if we compute disparity maps in both directions (left-to-right and right-to-left), a correctly matched pixel should be consistent in both directions.

For a pixel at position $(x, y)$ in the left image with disparity $d$, the corresponding pixel in the right image is at $(x - d, y)$. If we look up the right-to-left disparity at that position, it should also be approximately $d$. Pixels that fail this check are typically in occluded regions or areas where the matching was unreliable, and are marked as invalid. This project fills the invalidated holes using a scanline-based approach that substitutes the smaller of the two nearest valid disparities on either side, which corresponds to the conservative assumption that occluded regions belong to the background (farther surface).

<br><br>

# Implementation Overview

This implementation of Semi-Global Matching is written from scratch in Python using NumPy and SciPy, without relying on OpenCV or any other computer vision library for the core stereo matching logic. The stereo image pairs used in this implementation are from the [Middlebury Stereo Datasets](https://vision.middlebury.edu/stereo/data/).

**Key implementation features:**
- **Census transform** with uint64 encoding supporting kernels up to 8×8 (64 bits)
- **Semi-Global Matching** with four-path cost aggregation (north, south, east, west)
- **Subpixel refinement** using parabolic interpolation for sub-integer disparity precision
- **Left-right consistency check** by computing both LR and RL disparity maps and cross-validating
- **Scanline hole filling** for occluded regions using the background (minimum disparity) assumption
- **3D point cloud generation** with proper pinhole camera reprojection and PLY export
- **Quantitative evaluation** against ground truth with EPE, Bad Pixel %, RMSE, and MAE metrics

### Usage

**Step 1: Configure parameters and run**

Parameters are defined within `main.py`:
- **data_path**: Directory containing the stereo pair
- **max_disparity**: Maximum disparity search range in pixels
- **kernel_size**: Census transform window size (must satisfy kernel_size² ≤ 64)
- **penalty1 / penalty2**: SGM smoothness penalties (P1 < P2)
- **focal_length / baseline**: Camera parameters for 3D reprojection (from dataset documentation)
- **lr_consistency_threshold**: Maximum allowed disparity difference for LR check (in pixels)

```
uv run main.py
```

**Output files:**
- **Disparity map**: `output/dolls_sgm.png` — jet colormap visualization of the final disparity
- **Point cloud**: `output/dolls_sgm.ply` — 3D point cloud viewable in Open3D or MeshLab
- **Console output**: Per-stage diagnostics and quantitative metrics against ground truth


**Step 2: To visualize the 3D point cloud:**
```
uv run visualize.py output/dolls_sgm.ply
```

### Results

```
GT: dtype = uint8, range = [0, 222], non-zero = 1527119
  LR disparity time = 95.69s, range = [0.0, 239.0]
  RL disparity time = 95.87s, range = [0.0, 239.0]
  After LR check: valid pixels = 78.8%, range = [0.0, 216.7]
  After hole fill: valid pixels = 99.5%, range = [0.0, 216.7]
  Filtered range: [0.0, 215.8]

Metrics vs GT:
  Raw LR only:  {'EPE': 11.54, 'Bad Pixel %': 17.11, 'RMSE': 36.16, 'MAE': 11.54}
  After LR check (valid pixels only): {'EPE': 0.96, 'Bad Pixel %': 3.2, 'RMSE': 3.36, 'MAE': 0.96}
  After filter   (valid pixels only): {'EPE': 0.89, 'Bad Pixel %': 2.96, 'RMSE': 3.08, 'MAE': 0.89}
```

**Note:** This implementation prioritizes educational clarity and correctness. The pure Python loops in the census transform and subpixel refinement make it significantly slower than optimized C++ implementations.
