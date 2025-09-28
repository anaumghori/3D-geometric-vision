# Stereo Vision Distance Measurement

### Table of Contents
- [What is Stereo Vision?](#what-is-stereo-vision)
- [Stereo Vision Fundamentals](#stereo-vision-fundamentals)
  - [Camera Calibration](#camera-calibration)
  - [Stereo Rectification](#stereo-rectification)
  - [Disparity Estimation](#disparity-estimation)
  - [Depth Calculation](#depth-calculation)
  - [Stereo Geometry](#stereo-geometry)
  - [Disparity-to-Depth Relationship](#disparity-to-depth-relationship)
  - [Triangulation Principle](#triangulation-principle)
- [Implementation Overview](#implementation-overview)
  - [Usage](#usage)

<br><br>

# What is Stereo Vision?
Stereo vision is a computer vision technique that mimics human binocular vision to perceive depth and measure distances. Just like our eyes, which use two slightly different viewpoints to judge how far away objects are, stereo vision systems use two cameras positioned at a known distance apart. By comparing the images from these cameras, we can estimate the 3D positions of objects in a scene.

**How Distance Measurement Works**  
The core idea behind stereo vision is triangulation. Each camera captures an image of the same scene from a slightly different viewpoint. As a result, objects appear at different horizontal positions in the two images, and this difference is called disparity. The disparity is inversely proportional to the distance: objects that are closer to the cameras have larger disparities, while those farther away appear with smaller disparities. By combining the known distance between the cameras with the measured disparity, the system can accurately calculate the distance to each object in the scene.  

<br><br>

https://github.com/user-attachments/assets/fcea31b8-97a4-4651-9f13-c8c3e0314e76

<br><br>

# Stereo Vision Fundamentals

### Camera Calibration  
Camera calibration for stereo vision extends the single-camera calibration process (detailed in [Camera Calibration documentation](https://github.com/anaumghori/3D-geometric-vision/blob/main/CameraCalibration/README.md)) to determine the geometric relationship between two cameras. While single-camera calibration finds intrinsic parameters like focal length and lens distortion, stereo calibration additionally computes the **relative pose** between cameras, including their relative rotation and translation.

The stereo calibration process uses the [same chessboard patterns](https://github.com/anaumghori/3D-geometric-vision/blob/main/CameraCalibration/README.md#step-1-chessboard-corner-detection) captured simultaneously by both cameras. The key output is the **baseline**; the physical distance between the two camera centers, which is essential for depth calculation. The calibration also determines how the two cameras are oriented relative to each other, accounting for any misalignment in their mounting.

**Essential stereo calibration parameters:**
- **Intrinsic matrices** for both cameras (focal lengths, principal points, distortion coefficients)
- **Rotation matrix R**: Describes how the right camera is rotated relative to the left
- **Translation vector t**: Gives the 3D offset between camera centers
- **Baseline B**: The magnitude of the translation vector, typically the horizontal separation

<br><br>

### Stereo Rectification
Stereo rectification is a preprocessing step that transforms both stereo images so that corresponding points lie on the same horizontal scanlines. This simplification reduces the 2D correspondence problem to a 1D search along horizontal lines, dramatically improving both the speed and reliability of disparity estimation.

Without rectification, corresponding points can appear anywhere in the second image, requiring a computationally expensive 2D search. After rectification, the search is constrained to the same horizontal row, making disparity estimation much more efficient and less prone to errors. The rectification process involves computing homography-like transformations that align both image planes with the baseline. For more details on homography mathematics and robust estimation techniques like RANSAC, see [Panorama Stitching documentation](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/README.md).

**The rectification process involves:**
1. **Computing rectification transforms**: Calculate rotation matrices that align both image planes with the baseline
2. **Image warping**: Apply these transforms to both images, creating aligned stereo pairs
3. **Epipolar constraint enforcement**: Ensure that corresponding points have the same y-coordinate

**Key benefits of rectification:**
- **Simplified matching**: Corresponding points lie on the same horizontal line
- **Improved accuracy**: Reduces false matches and matching ambiguities
- **Computational efficiency**: Enables faster stereo matching algorithms
- **Algorithm simplification**: Standard stereo algorithms assume rectified inputs

After rectification, the epipolar lines become horizontal, which is why disparity search becomes a 1D problem along image rows. The rectification process may introduce slight image distortions, but these are typically minimal and do not affect the accuracy of distance measurements. 

<br><br>

### Disparity Estimation
Disparity estimation is the core of stereo vision. It involves finding corresponding pixels between the left and right images and measuring their horizontal displacement. The resulting disparity values form the basis for calculating object distances.

#### Semi-Global Block Matching (SGBM)

A widely used method for disparity estimation is **Semi-Global Block Matching (SGBM)**. At its core, the method starts with **block matching**, where small image patches (or blocks) from the left image are compared with corresponding patches in the right image to find the best match. For each possible disparity value, a **matching cost** is computed, which represents how similar the two patches are. Instead of relying only on local information, SGBM introduces a **global optimization** step that applies smoothness constraints across the image. This ensures that the resulting disparity map is both coherent and less noisy. Finally, the algorithm performs **sub-pixel refinement**, interpolating between discrete disparity values to achieve higher precision in distance estimation.

The performance of SGBM depends on several key parameters. 
1. The **block size** determines how large each comparison window is: larger blocks provide more stability but may lose fine details, while smaller blocks capture detail but are more sensitive to noise. 
2. The **disparity range** sets the maximum expected disparity, which is tied to the depth of the scene and the distance between the cameras. 
3. The **penalty parameters** are used to balance smoothness and detail: they discourage sudden disparity changes unless strongly supported by the image, preventing overly jagged or inconsistent results.

Despite its strengths, disparity estimation faces challenges. **Textureless regions** (like blank walls) provide little information for matching, while **occlusions** create areas where objects are visible in one image but not the other. **Repetitive patterns** can mislead the algorithm into incorrect matches, and **lighting differences** between the cameras may affect consistency.

The final output of SGBM is a **disparity map** that encodes the horizontal shift for every pixel. Higher disparity values correspond to closer objects, while lower values represent distant ones. Pixels without reliable matches are marked as invalid and require special handling in later processing steps.

<br><br>

| Example 1 | Example 2 |
|----------|----------|
| ![Example 1](https://github.com/user-attachments/assets/291a99ad-41d1-4f23-b6da-2fc51306b0b5)         | ![Example 2](https://github.com/user-attachments/assets/463f6246-07cc-4817-a319-de0886dd7ca5)         |

<br><br>

### Depth Calculation
Depth calculation transforms the disparity map into real-world distance measurements using the principles of stereo geometry. In this step, the pixel-based disparity values are converted into metric distances by applying a geometric relationship derived from similar triangles. The formula is given by:

$$Z = \frac{f \cdot B}{d}$$

where $Z$ is the depth (the distance from the cameras to the object), $f$ is the focal length in pixels obtained through camera calibration, $B$ is the baseline or distance between the two cameras, and $d$ is the disparity in pixels determined during stereo matching.

This relationship highlights the **inverse connection between depth and disparity**: closer objects produce high disparity values and appear very different between the two views, while distant objects yield low disparity values and look nearly identical. In the theoretical case of infinite distance, disparity becomes zero, meaning the object appears at the same position in both images.

In practice, several factors influence the accuracy of depth calculation: 
1. The **disparity resolution** sets the smallest measurable disparity step, which limits the maximum detectable distance. 
2. The **baseline** introduces a trade-off: a larger baseline improves depth precision but reduces the common field of view between cameras. 
3. The **focal length** affects the measurement range: longer focal lengths increase accuracy but narrow the field of view. 

Accurate depth calculation assumes that the stereo system is well calibrated and that the disparity map is reliable. Even small disparity errors can result in large depth inaccuracies, particularly for distant objects where small pixel shifts correspond to substantial changes in distance.

<br><br>

| Example 1 | Example 2 |
|----------|----------|
| ![Example 1](https://github.com/user-attachments/assets/aa4b10ff-dd61-4264-85dc-4e80d1892610)         | ![Example 2](https://github.com/user-attachments/assets/9022061a-d6d3-48eb-8e71-90c67fe6f18d)         |

<br><br>

### Stereo Geometry
Stereo geometry defines the mathematical relationship between two cameras and explains how 3D points in the real world are projected onto their respective image planes. This relationship is the foundation of stereo vision, as it allows us to convert 2D pixel measurements from images into 3D world coordinates. 

Stereo vision involves several coordinate systems:
- **World coordinates**: 3D points in the real world $(X, Y, Z)$
- **Left camera coordinates**: 3D points relative to the left camera
- **Right camera coordinates**: 3D points relative to the right camera  
- **Image coordinates**: 2D pixel locations $(u, v)$ in each camera

**Camera projection model**

Each camera follows the pinhole projection model discussed in our [Camera Calibration documentation](https://github.com/anaumghori/3D-geometric-vision/blob/main/CameraCalibration/README.md). A 3D point $\mathbf{P} = [X, Y, Z]^T$ projects to image coordinates through:

$$\begin{bmatrix} u \\\ v \\\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} X/Z \\\ Y/Z \\\ 1 \end{bmatrix}$$

Where $\mathbf{K}$ is the camera intrinsic matrix containing focal length and principal point information.

**Stereo camera relationship**

In a calibrated stereo system, the relationship between left and right cameras is described by:
- **Rotation matrix R**: $3 \times 3$ matrix describing relative orientation
- **Translation vector t**: $3 \times 1$ vector describing relative position
- **Baseline B**: Horizontal separation between cameras (usually $B = |t_x|$)

A point in left camera coordinates $\mathbf{P}_L$ relates to right camera coordinates $\mathbf{P}_R$ through:

$$\mathbf{P}_R = \mathbf{R} \mathbf{P}_L + \mathbf{t}$$

**Epipolar geometry**

The geometric constraint that corresponding points must satisfy is called the **epipolar constraint**. For rectified stereo images, this constraint simplifies to the requirement that corresponding points lie on the same horizontal scanline, making disparity search a 1D problem along image rows.

<br><br>

### Disparity-to-Depth Relationship

The core mathematical relationship in stereo vision connects the observed disparity between corresponding points to their real-world depth. This relationship emerges from the geometric properties of the stereo camera setup and forms the foundation for all distance measurements.

**Deriving the depth formula**

Consider a 3D point $\mathbf{P} = [X, Y, Z]^T$ observed by both cameras. After rectification, this point projects to:
- **Left image**: $x_L = f \cdot \frac{X}{Z}$  
- **Right image**: $x_R = f \cdot \frac{X - B}{Z}$

Where $f$ is the focal length and $B$ is the baseline distance.

The **disparity** is defined as the horizontal difference between these projections:

$$d = x_L - x_R = f \cdot \frac{X}{Z} - f \cdot \frac{X - B}{Z}$$

Simplifying this expression:

$$d = f \cdot \frac{X - (X - B)}{Z} = f \cdot \frac{B}{Z}$$

Solving for depth $Z$:

$$Z = \frac{f \cdot B}{d}$$

**Physical interpretation**

This fundamental relationship reveals several important properties:
- **Inverse relationship**: Depth is inversely proportional to disparity
- **Resolution limits**: Small disparities (distant objects) have poor depth resolution
- **Baseline effects**: Larger baselines provide better depth discrimination
- **Focal length scaling**: Longer focal lengths improve depth precision

**Measurement precision**

The depth measurement precision depends on the disparity precision. A small error $\delta d$ in disparity causes a depth error:

$$\delta Z = \frac{f \cdot B}{d^2} \cdot \delta d$$

This shows that depth errors grow quadratically with distance, meaning distant objects have much larger absolute distance uncertainties than nearby objects.

<br><br>

### Triangulation Principle
Triangulation is the geometric process of determining the 3D position of a point by measuring angles from two known locations. In stereo vision, this principle enables the reconstruction of 3D scene geometry from 2D image observations.

Given two cameras with known positions and orientations, and corresponding image points, triangulation finds the 3D point that projects to both observed locations. Geometrically, this involves finding the intersection of two rays:
- **Ray 1**: From left camera center through the observed point in the left image
- **Ray 2**: From right camera center through the observed point in the right image

**Mathematical formulation**

For rectified stereo cameras, triangulation simplifies significantly. The 3D coordinates of a point can be computed directly from the disparity:

$$X = \frac{(u_L - c_x) \cdot Z}{f} = \frac{(u_L - c_x) \cdot B}{d}$$

$$Y = \frac{(v_L - c_y) \cdot Z}{f} = \frac{(v_L - c_y) \cdot B}{d}$$  

$$Z = \frac{f \cdot B}{d}$$

Where:
- $(u_L, v_L)$ are the pixel coordinates in the left image
- $(c_x, c_y)$ is the principal point of the left camera
- $f$ is the focal length, $B$ is the baseline, $d$ is the disparity

**Triangulation accuracy factors**

The accuracy of triangulation depends on several geometric factors:
- **Intersection angle**: Larger angles between rays provide better triangulation
- **Baseline length**: Longer baselines improve depth resolution but reduce overlap
- **Distance to object**: Nearby objects triangulate more accurately than distant ones
- **Measurement precision**: Subpixel accuracy in feature detection improves 3D precision

**Error propagation and practical considerations**

Small errors in image measurements propagate to 3D position errors. The sensitivity is highest for distant objects and decreases with closer proximity. This relationship determines the practical operating range for stereo vision systems and influences design decisions about baseline and focal length selection.

In practice, the rays rarely intersect perfectly due to noise, so least-squares solutions are typically used to find the best 3D point that minimizes the sum of squared distances to both rays. This approach provides robust triangulation even in the presence of measurement uncertainties.

<br><br>

# Implementation Overview
This implementation demonstrates stereo vision distance measurement using Python, OpenCV, and YOLO object detection. The system processes synchronized stereo image sequences to detect vehicles and measure their distances in real-time. 

The implementation consists of two main files: `main.py` contains the high-level pipeline orchestration and video processing logic, while `utils.py` provides specialized functions for calibration loading, stereo processing, object detection, and visualization. 

### Usage
Because the stereo image dataset is quite large, it is not included in this repository. To set it up, download the dataset from either of the following links:  
- [The KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/raw_data.php)  
- [Kitti Dataset on Kaggle](https://www.kaggle.com/datasets/klemenko/kitti-dataset)

and format it into a directory structure like the following:

```
data/
│
├── Set1/                    # Left camera images
│   ├── 0000000000.png
│   ├── 0000000001.png
│   └── ...
│
├── Set2/                    # Right camera images  
│   ├── 0000000000.png
│   ├── 0000000001.png
│   └── ...
│
└── calib_cam_to_cam.txt     # KITTI calibration file
```

**Running the system:**
```
uv run main.py
```

**Output files:**
- **output.mp4**: Annotated video showing detected vehicles with distance measurements
- **disparity_maps/**: Color-coded disparity visualizations for the first few frames
- **depth_maps/**: Color-coded depth visualizations showing distance information
- **Console output**: Real-time processing statistics and system parameters