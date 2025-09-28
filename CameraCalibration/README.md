# Camera Calibration Using Zhang's Method

### Table of Contents
- [What is Camera Calibration?](#what-is-camera-calibration)
- [The Pinhole Camera Model](#the-pinhole-camera-model)
- [Camera Calibration Steps](#camera-calibration-steps)
  - [Step 1: Chessboard Corner Detection](#step-1-chessboard-corner-detection)
  - [Step 2: Homography Estimation](#step-2-homography-estimation)
  - [Step 3: Intrinsic Parameter Estimation](#step-3-intrinsic-parameter-estimation)
  - [Step 4: Extrinsic Parameter Calculation](#step-4-extrinsic-parameter-calculation)
  - [Step 5: Non-linear Refinement](#step-5-non-linear-refinement)
  - [Step 6: Distortion Correction](#step-6-distortion-correction)
- [Implementation Overview](#implementation-overview)
  - [Usage](#usage)

<br><br>

# What is Camera Calibration?

Camera calibration is the process of determining the internal geometric and optical characteristics of a camera system. Think of it as creating a precise mathematical model that describes how your camera transforms the 3D world into 2D images. Just as you might calibrate a scale to ensure accurate measurements, camera calibration ensures that your camera produces geometrically accurate representations of the real world.

The fundamental question camera calibration answers is: **"Given a 3D point in the world, where will it appear in my camera's image, and vice versa?"** This relationship is crucial for applications like:
- **3D reconstruction**: Building 3D models from 2D images
- **Augmented reality**: Overlaying virtual objects onto real scenes  
- **Robot vision**: Enabling robots to understand spatial relationships
- **Stereo vision**: Measuring depth and distance from multiple camera views
- **Image rectification**: Removing distortions caused by camera lenses

Camera calibration determines two types of parameters:
1. **Intrinsic parameters**: Properties internal to the camera (focal length, optical center, distortion)
2. **Extrinsic parameters**: The camera's position and orientation in 3D space relative to the calibration pattern

Real cameras deviate from ideal mathematical models in several ways:
- **Lens distortion**: Straight lines in the world appear curved in images, especially near image borders
- **Manufacturing variations**: No two cameras are identical, even from the same model
- **Optical imperfections**: Lenses introduce various aberrations and distortions
- **Unknown focal length**: The effective focal length varies with focus settings and manufacturing tolerances

Without calibration, measurements from images would be inaccurate, 3D reconstruction would fail, and computer vision algorithms would produce unreliable results.

<br><br>

| Calibration Result 1 | Calibration Result 2 |
|----------|----------|
| ![Result 1](https://github.com/anaumghori/3D-geometric-vision/blob/main/CameraCalibration/results/undistorted_3.png) | ![Result 2](https://github.com/anaumghori/3D-geometric-vision/blob/main/CameraCalibration/results/undistorted_20.png) |

<br><br>

# The Pinhole Camera Model

The pinhole camera model is the fundamental mathematical framework for understanding how 3D points project onto 2D images. Imagine a box with a tiny hole in one side, light rays from the outside world pass through this hole and create an inverted image on the opposite wall. This simple concept forms the basis for all camera calibration mathematics. 

In the pinhole model, every 3D point in the world projects through a single point (the camera center) onto the image plane. This creates the perspective effect we see in photographs; objects farther away appear smaller, and parallel lines converge to vanishing points. Every visible point in the 3D world reflects light in all directions, but the tiny pinhole acts as a selective filter, allowing exactly one ray from each point to pass through. All other rays are blocked by the camera walls, ensuring that each world point contributes to exactly one pixel location. This process happens simultaneously for all visible points, with millions of rays passing through the same tiny opening at once, each following its own geometric path to create a complete, sharp image. The resulting image is both horizontally and vertically inverted because rays from the top of objects travel downward through the pinhole to hit the bottom of the image plane, while rays from the left side of objects end up on the right side of the image, creating the characteristic flipped appearance that our brains (or camera software) must correct.

A 3D world point $\mathbf{P} = [X, Y, Z]$ projects to a 2D image point $\mathbf{p} = [x, y]$ according to:

$\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad x = f \cdot \frac{X}{Z}, \quad \quad y = f \cdot \frac{Y}{Z}$

Where **f** is the focal length, and **Z** represents the depth (distance from the camera).

### Key Assumptions of the Pinhole Model

1. **Single projection center**: All light rays pass through one point
2. **Linear perspective**: Straight lines remain straight (before distortion)
3. **No lens effects**: The ideal model ignores lens distortion, depth of field, and other optical effects
4. **Instantaneous capture**: No motion blur or temporal effects

### From World to Pixels: The Complete Pipeline

The journey from a 3D world point to a pixel location involves several coordinate transformations:

1. **World coordinates → Camera coordinates**: Apply rotation and translation based on camera pose
2. **Camera coordinates → Normalized image coordinates**: Apply perspective division (divide by Z)
3. **Normalized coordinates → Pixel coordinates**: Apply focal length and principal point transformation
4. **Ideal pixels → Distorted pixels**: Apply lens distortion model

This pipeline is mathematically expressed as:

$$\begin{bmatrix} u \\\ v \\\ 1 \end{bmatrix} 
= \mathbf{K} \begin{bmatrix} \mathbf{R} & \mathbf{t} \end{bmatrix} 
\begin{bmatrix} X \\\ Y \\\ Z \\\ 1 \end{bmatrix}$$

Where:
- $\mathbf{K}$ is the intrinsic matrix (internal camera parameters)
- $[\mathbf{R} \mid \mathbf{t}]$ is the extrinsic matrix (camera pose in world coordinates)
- $[X, Y, Z, 1]^T$ is the 3D world point in homogeneous coordinates
- $[u, v, 1]^T$ is the resulting pixel location in homogeneous coordinates

<br><br>

# Camera Calibration Steps

### Step 1: Chessboard Corner Detection

Chessboard corner detection is the foundation of camera calibration, providing the precise 2D–3D point correspondences needed to solve for camera parameters. A chessboard pattern is ideal because its corners are well-defined geometric features that can be detected with subpixel accuracy. The alternating black and white squares create strong gradients at corner locations, making them easy for computer vision algorithms to identify reliably.

The process begins with automatic corner detection using algorithms like the Harris corner detector or specialized chessboard detection methods. However, initial detection typically achieves only pixel-level accuracy: the algorithm might say a corner lies exactly at pixel coordinates (245, 156), when in reality the true corner position could be slightly offset - say, (245.3, 156.7). This small difference may not sound important, but in camera calibration, even a 0.3-pixel error can propagate through the 3D reconstruction process and cause significant inaccuracies.

To address this, **subpixel refinement** is applied. Subpixel accuracy means estimating corner positions more precisely than the pixel grid allows, at fractional pixel locations. The refinement process starts from the rough corner location detected by the algorithm, then analyzes the brightness patterns in the surrounding neighborhood. By examining how the image gradients intersect, the algorithm identifies the true geometric center of the corner. The refined result might be, for example, (245.27, 156.73), which is far more accurate than the initial whole-pixel guess.

**Why chessboards work so well:**
1. **High contrast**: Black and white squares provide strong, consistent gradients
2. **Known geometry**: The pattern has precisely known dimensions and structure
3. **Rich information**: Many corners provide abundant constraints for parameter estimation
4. **Unambiguous detection**: The regular pattern helps avoid false matches

Finally, the algorithm establishes point correspondences by pairing each detected 2D corner in the image with its corresponding 3D location on the physical chessboard. For example, if we know the chessboard has 30 mm squares, then corner positions in the real world are precisely known (e.g., [0, 0, 0], [30, 0, 0], [60, 0, 0], etc.). These accurate 2D–3D matches form the basis for solving the camera's intrinsic and extrinsic parameters.

<br><br>

### Step 2: Homography Estimation

For each calibration image, we compute a **homography matrix** that maps 2D chessboard coordinates to 2D image coordinates. Since the chessboard is planar (flat), this 2D-to-2D transformation can be described by a 3×3 homography matrix **H**. This is essentially a "flattened" version of the full 3D camera projection, we're temporarily ignoring the third dimension since all points lie on a plane.

The homography relates world plane coordinates $[X, Y]$ to image coordinates $[u, v]$ through:

$$\lambda \begin{bmatrix} u \\\ v \\\ 1 \end{bmatrix} 
= \mathbf{H} \begin{bmatrix} X \\\ Y \\\ 1 \end{bmatrix}$$

**The key insight**: Although we're dealing with 3D-to-2D camera projection, the constraint that all world points lie on a plane reduces the problem to a simpler 2D-to-2D transformation. This homography encodes information about both the camera's intrinsic parameters and its extrinsic pose relative to the chessboard.

**Computing the Homography**

Each point correspondence between world coordinates $(X_i, Y_i)$ and image coordinates $(u_i, v_i)$ provides two linear equations. Writing the homography as:

$$\mathbf{H} = \begin{bmatrix} 
h_{11} & h_{12} & h_{13} \\\ 
h_{21} & h_{22} & h_{23} \\\ 
h_{31} & h_{32} & h_{33} 
\end{bmatrix}$$

The projection equation becomes:

$$\begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix} \sim \begin{bmatrix} h_{11}X_i + h_{12}Y_i + h_{13} \\ h_{21}X_i + h_{22}Y_i + h_{23} \\ h_{31}X_i + h_{32}Y_i + h_{33} \end{bmatrix}$$

Since this is a homogeneous equation (the $\sim$ symbol indicates equality up to scale), we can write:

$$u_i = \frac{h_{11}X_i + h_{12}Y_i + h_{13}}{h_{31}X_i + h_{32}Y_i + h_{33}}$$

$$v_i = \frac{h_{21}X_i + h_{22}Y_i + h_{23}}{h_{31}X_i + h_{32}Y_i + h_{33}}$$

Cross-multiplying to eliminate fractions gives us two linear equations per point:

$$h_{11}X_i + h_{12}Y_i + h_{13} - u_i(h_{31}X_i + h_{32}Y_i + h_{33}) = 0$$

$$h_{21}X_i + h_{22}Y_i + h_{23} - v_i(h_{31}X_i + h_{32}Y_i + h_{33}) = 0$$

**Linear System Formulation**

The homography matrix has 9 parameters, but due to the scale ambiguity inherent in homogeneous coordinates, we have only 8 degrees of freedom. With N point correspondences, we get 2N equations in these 8 unknown parameters. This creates a homogeneous linear system:

$$\mathbf{A}\mathbf{h} = \mathbf{0}$$

where $\mathbf{h} = [h_{11}, h_{12}, h_{13}, h_{21}, h_{22}, h_{23}, h_{31}, h_{32}, h_{33}]^T$ and each point correspondence contributes two rows to matrix $\mathbf{A}$:

$$\begin{bmatrix} 
X_i & Y_i & 1 & 0 & 0 & 0 & -u_iX_i & -u_iY_i & -u_i \\\ 
0 & 0 & 0 & X_i & Y_i & 1 & -v_iX_i & -v_iY_i & -v_i 
\end{bmatrix}$$

Since we need at least 4 point correspondences to solve for the 8 degrees of freedom, and in practice we use many more points (typically 50-100 corners), this becomes an overdetermined system. We solve it using **Singular Value Decomposition (SVD)**, which finds the homography parameters as the right singular vector corresponding to the smallest singular value of matrix **A**.

<br><br>

### Step 3: Intrinsic Parameter Estimation

Zhang's method cleverly extracts intrinsic camera parameters from multiple chessboard images using homographies and linear algebra. To understand this process, let's first examine what we're trying to find.

**The Intrinsic Matrix Structure**

The intrinsic matrix **K** encodes the camera's internal geometric properties and transforms normalized camera coordinates to pixel coordinates:

$$\mathbf{K} = \begin{bmatrix} 
f_x & s & c_x \\\ 
0 & f_y & c_y \\\ 
0 & 0 & 1 
\end{bmatrix}$$

Each parameter has a specific physical meaning:

- **fx, fy (focal lengths)**: Measured in pixels, these represent how much the camera "zooms" in x and y directions. They're related to the physical focal length f and pixel dimensions by fx = f/pixel_width and fy = f/pixel_height. A larger focal length means a narrower field of view and greater magnification of distant objects.  

- **cx, cy (principal point)**: The pixel coordinates where the optical axis intersects the image plane. Ideally at the image center, but manufacturing imperfections often cause slight offsets.  

- **s (skew coefficient)**: Accounts for non-rectangular pixel shapes, which can occur if the image sensor isn't perfectly aligned with the optical system. For most modern cameras, this value is close to zero.  

**Zhang's Extraction Strategy**

The key insight is that each homography, which maps a 3D plane to its 2D image projection, is linked by the same intrinsic matrix K. By analyzing these relationships across different images, we can solve for the intrinsic parameters without knowing the exact camera poses. Mathematically, Zhang exploits the fact that the first two columns of each homography, when transformed by **K**, correspond to the first two columns of a rotation matrix, which must be orthonormal. Think of it like this: when you take a photo of a chessboard from different angles, the geometric relationships within each image must satisfy certain mathematical constraints because they all come from the same camera with fixed internal properties. This relationship can be expressed through the homography decomposition:

$$\mathbf{H} = \lambda \mathbf{K} [\mathbf{r_1}, \mathbf{r_2}, \mathbf{t}]$$

Where **H** is the observed homography, λ is a scale factor, **K** is the intrinsic matrix, and [**r₁**, **r₂**, **t**] represents the first two columns of the rotation matrix and the translation vector. Since **r₁** and **r₂** are columns of a rotation matrix, they must satisfy strict orthogonality conditions:

1. **Orthogonality constraint**: Since **r₁** and **r₂** are columns of a rotation matrix, they must be orthogonal and have unit length:
   
   $$\mathbf{r_1}^T \mathbf{r_2} = 0 \quad \text{(orthogonal)}$$
   
   $$\mathbf{r_1}^T \mathbf{r_1} = \mathbf{r_2}^T \mathbf{r_2} = 1 \quad \text{(unit length)}$$

3. **Intrinsic relationship**: These constraints can be expressed in terms of the homography columns **h₁** and **h₂**:
   
   $$\mathbf{h_1}^T \mathbf{K}^{-T} \mathbf{K}^{-1} \mathbf{h_2} = 0$$
   
   $$\mathbf{h_1}^T \mathbf{K}^{-T} \mathbf{K}^{-1} \mathbf{h_1} = \mathbf{h_2}^T \mathbf{K}^{-T} \mathbf{K}^{-1} \mathbf{h_2}$$

4. **Linear system formulation**: By defining **B** = **K⁻ᵀK⁻¹** and parameterizing it appropriately, these constraints become linear in the unknown intrinsic parameters.

**The SVD Solution Process**

Each chessboard image provides multiple equations through these orthogonality constraints, creating a classic overdetermined system: 10 images might give us 20 equations, but we only need to solve for 5 parameters. In a perfect noise-free world, every equation would agree exactly on the same parameter values. In practice, however, image measurements have small errors, so no single solution satisfies every equation perfectly.

This is where Singular Value Decomposition (SVD) becomes essential. Acting as a "best fit" solver, SVD takes the slightly inconsistent equations from all images and finds the solution that best satisfies them collectively while minimizing the effect of noise. Since Zhang's formulation is a homogeneous linear system, SVD is particularly well-suited because it's robust, stable, and naturally handles noisy, overdetermined problems.

**Parameter Extraction from the B Matrix**

Once we solve for the B matrix using SVD, we extract the intrinsic parameters using the following formulas:

$$u_y = \frac{B_{12}B_{13} - B_{11}B_{23}}{B_{11}B_{22} - B_{12}^2}$$

$$\lambda = B_{33} - \frac{B_{13}^2 + u_y(B_{12}B_{13} - B_{11}B_{23})}{B_{11}}$$

$$\alpha_x = \sqrt{\frac{\lambda}{B_{11}}}$$

$$\alpha_y = \sqrt{\frac{\lambda B_{11}}{B_{11}B_{22} - B_{12}^2}}$$

$$s = -\frac{B_{12}\alpha_x^2\alpha_y}{\lambda}$$

$$u_x = \frac{s \cdot u_y}{\alpha_y} - \frac{B_{13}\alpha_x^2}{\lambda}$$

Where the intrinsic parameters are:
- **fx = αx, fy = αy**: focal lengths in pixels
- **cx = ux, cy = uy**: principal point coordinates  
- **s**: skew coefficient

**The Complete Linear Algebra Approach:**

1. **Constraint formulation**: Express orthogonality conditions from homographies as linear equations in the intrinsic parameters
2. **System assembly**: Stack constraints from multiple images into a single matrix equation
3. **SVD solution**: Use SVD to find the best-fit solution that works across all images
4. **Parameter extraction**: Convert the B matrix solution into the physical intrinsic parameters using the formulas above

<br><br>

### Step 4: Extrinsic Parameter Calculation

Once we know the intrinsic parameters **K**, we can calculate the extrinsic parameters (camera pose) for each calibration image. The extrinsic parameters answer two questions: where the camera was positioned and how it was oriented when each image was captured. This is essential because during calibration, the camera (or chessboard) is moved around, and for each image, the camera has a different pose relative to the chessboard.

**Computing Extrinsic Parameters**

For each homography $\mathbf{H}$, the extrinsic parameters are extracted using:

$$\mathbf{H} = \mathbf{K} [\mathbf{r_1}, \mathbf{r_2}, \mathbf{t}]$$

The specific computation involves:

$$\mathbf{r_1} = \mu \mathbf{K}^{-1} \mathbf{h_1}$$

$$\mathbf{r_2} = \mu \mathbf{K}^{-1} \mathbf{h_2}$$

$$\mathbf{r_3} = \mathbf{r_1} \times \mathbf{r_2}$$

$$\mathbf{t} = \mu \mathbf{K}^{-1} \mathbf{h_3}$$

where the scale factor μ is computed as:

$$\mu = \frac{1}{||\mathbf{K}^{-1} \mathbf{h_1}||}$$

Here, **h₁**, **h₂**, and **h₃** are the first, second, and third columns of the homography matrix **H** respectively.

**Physical Interpretation**

The rotation matrix $\mathbf{R} = [\mathbf{r_1}, \mathbf{r_2}, \mathbf{r_3}]$ tells us how the camera was tilted or rotated (roll, pitch, yaw), while the translation vector $\mathbf{t}$ tells us where the camera was located (left/right, up/down, forward/back). Together, $[\mathbf{R}|\mathbf{t}]$ gives the complete 6 degrees of freedom of the camera pose for each calibration photo.

Intuitively, this step reconstructs the path of your calibration session: if you walked around a chessboard and took ten photos, the extrinsic parameters reveal exactly where you were standing and which way the camera was facing for each shot. This not only recovers the geometry of the setup but also provides a useful check; if the recovered camera positions look realistic, the calibration likely worked well, if they look impossible (e.g., the camera appears inside the chessboard), then something went wrong.

<br><br>

### Step 5: Non-linear Refinement

The linear solution from Zhang's method provides an excellent starting point, but achieving the highest accuracy requires non-linear optimization. Real cameras exhibit non-linear effects that the linear approach cannot capture, most significantly **lens distortion**, which causes straight lines to appear curved in images. Additionally, noise in corner detection and limitations of the linear approximation introduce errors that can be reduced through refinement.

**Iterative Distortion Refinement Process**

Zhang's method includes a specific iterative algorithm to handle distortion during the linear estimation phase:

1. **Initial linear estimation**: Compute the linear parameters ignoring radial distortion (Steps 1-4)

2. **Distortion computation**: If we project the 3D points into the image using the linear model, deviations will still be present between the projected and extracted points due to radial distortion. From the linear parameters and the 3D points, we compute $(n_x, n_y)$ using:

$$n_x = \frac{m_{ux}}{m_{ox}}, \quad n_y = \frac{m_{uy}}{m_{oy}}$$

where $(m_{ux}, m_{uy})$ are the ideal undistorted coordinates and $(m_{ox}, m_{oy})$ are the observed coordinates.

3. **Linear constraint setup**: The radial distorted point is:
   
$$\begin{bmatrix} x \\\ y \\\ 1 \end{bmatrix} 
= \begin{bmatrix} 
\alpha_x n_x(1 + \kappa_1 r^2 + \kappa_2 r^4) + s n_y(1 + \kappa_1 r^2 + \kappa_2 r^4) + u_x \\\ 
\alpha_y n_y(1 + \kappa_1 r^2 + \kappa_2 r^4) + u_y \\\ 
1 
\end{bmatrix}$$
   
4. **Solve for distortion coefficients**: Stack all these equations into one system and solve for κ₁ and κ₂ parameters

5. **Radial undistortion**: Use κ₁ and κ₂ to radially undistort the 2D points and feed this again to the linear parameter estimation

6. **Iterate**: Repeat steps 2-5 until convergence

**Global Non-linear Optimization**

After the iterative distortion process, the final refinement step minimizes the **reprojection error** - the difference between observed feature locations and where the camera model predicts they should appear. The mathematical objective is:

$$\text{minimize} \sum_{i} \sum_{j} ||\mathbf{p}_{ij} - \text{project}(\mathbf{X}_{ij}, \mathbf{K}, \text{distortion}, \mathbf{R}_i, \mathbf{t}_i)||^2$$

Where:
- $\mathbf{p}_{ij}$ is the observed image location of point $j$ in image $i$
- $\mathbf{X}_{ij}$ is the corresponding 3D world point
- $\text{project}(...)$ applies the complete camera model (intrinsics + distortion + extrinsics)

**Levenberg-Marquardt Optimization**

The **Levenberg-Marquardt algorithm** is used for this optimization because it combines the stability of gradient descent with the fast convergence of Newton's method. It automatically adjusts the optimization strategy based on local conditions:
- When far from the minimum: acts like gradient descent (stable but slow)
- When near the minimum: acts like Newton's method (fast convergence)

All parameters are optimized simultaneously in what's called "bundle adjustment":
- **Intrinsic parameters**: focal lengths, principal point, skew
- **Distortion coefficients**: radial and tangential distortion parameters  
- **Extrinsic parameters**: 6 values per image (3 rotation + 3 translation)

This joint optimization accounts for correlations between parameters and typically reduces reprojection error by 30-50% compared to the linear solution alone.

**Convergence Criteria**

The optimization is iterative and continues until one of several convergence criteria is met:
- **Parameter change threshold**: optimization stops when parameter updates become very small
- **Error threshold**: optimization stops when the error reduction becomes negligible  
- **Maximum iterations**: prevents infinite loops in pathological cases (typically 10-50 iterations)

This refinement process transforms the good initial estimate from Zhang's linear method into a highly accurate camera calibration that accounts for real-world complexities.

<br><br>

### Step 6: Distortion Correction

Real camera lenses deviate from the ideal pinhole model, introducing various types of distortion. The most significant is **radial distortion**, caused by the spherical shape of lens elements. Light rays passing through different parts of the lens are bent by different amounts, causing straight lines to appear curved, particularly near image edges. **Barrel distortion** makes images appear to bulge outward (positive distortion coefficients), while **pincushion distortion** makes them appear pinched inward (negative distortion coefficients). The effects are most pronounced near image borders.

**Complete Distortion Model**

The distortion correction process involves several coordinate transformations:

1. **Convert to normalized coordinates**: First, convert pixel coordinates to normalized coordinates relative to the principal point:

$$x_{\text{norm}} = \frac{x_{\text{pixel}} - c_x}{f_x}$$

$$y_{\text{norm}} = \frac{y_{\text{pixel}} - c_y}{f_y}$$

2. **Calculate radial distance**: Compute the squared distance from the optical center in normalized coordinates:

$$r^2 = x_{\text{norm}}^2 + y_{\text{norm}}^2$$

3. **Apply radial distortion**: The radial distortion model modifies the normalized coordinates:

$$x_{\text{radial}} = x_{\text{norm}} \cdot (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

$$y_{\text{radial}} = y_{\text{norm}} \cdot (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$$

Where:
- $k_1, k_2, k_3$ are radial distortion coefficients
- Positive coefficients cause barrel distortion (outward bulging)  
- Negative coefficients cause pincushion distortion (inward pinching)

4. **Apply tangential distortion**: Tangential distortion occurs when the lens system is not perfectly centered or aligned with the image sensor:

$$x_{\text{distorted}} = x_{\text{radial}} + [2p_1 x_{\text{norm}} y_{\text{norm}} + p_2(r^2 + 2x_{\text{norm}}^2)]$$

$$y_{\text{distorted}} = y_{\text{radial}} + [p_1(r^2 + 2y_{\text{norm}}^2) + 2p_2 x_{\text{norm}} y_{\text{norm}}]$$

Where $p_1, p_2$ are tangential distortion coefficients, typically small for quality lenses.

5. **Convert back to pixel coordinates**: Finally, convert the distorted normalized coordinates back to pixel coordinates:

$$x_{\text{final}} = x_{\text{distorted}} \cdot f_x + c_x$$

$$y_{\text{final}} = y_{\text{distorted}} \cdot f_y + c_y$$

**Undistortion Process**

Once these distortion coefficients are estimated during calibration, they can be used to **undistort** images by applying the inverse transformation. Since the forward distortion equations are nonlinear, the inverse is typically computed iteratively or through lookup tables.

**Practical Considerations:**

- Most cameras require only **k₁** and **k₂** for adequate radial correction
- **k₃** is used for fisheye lenses or severe wide-angle distortion  
- Tangential distortion coefficients **p₁, p₂** are typically small for quality lenses
- Distortion effects are most pronounced near image borders
- The normalization step ensures distortion coefficients are independent of focal length and principal point values

<br><br>

This distortion correction capability is what makes Zhang's calibration method so valuable for applications requiring precise geometric measurements, such as 3D reconstruction, augmented reality, and robotic vision.

<br><br>

# Implementation Overview

This implementation of camera calibration follows Zhang's method using Python and OpenCV. The pipeline processes multiple images of a chessboard pattern to extract both intrinsic and extrinsic camera parameters, along with lens distortion coefficients. The implementation prioritizes educational clarity.

**Dataset source**: The calibration images used in this implementation are from the [Stereo Camera Chessboard Pictures dataset on Kaggle](https://www.kaggle.com/datasets/danielwe14/stereocamera-chessboard-pictures), which provides high-quality chessboard calibration images suitable for camera parameter estimation.

**Key implementation features:**
- **Automated corner detection** with subpixel refinement for maximum accuracy
- **Robust homography estimation** using RANSAC to handle potential outliers  
- **Zhang's method implementation** for linear intrinsic parameter estimation
- **Non-linear optimization** using Levenberg-Marquardt for parameter refinement
- **Comprehensive distortion modeling** including radial and tangential distortion
- **Quality assessment** through reprojection error analysis
- **Visual validation** by generating undistorted images with detected corners

**Technical considerations:**
- Requires at least 3 calibration images for reliable parameter estimation
- Chessboard pattern size and physical dimensions must be specified accurately
- Corner detection uses adaptive thresholding to handle varying lighting conditions

### Usage

Parameters, defined within the code itself:
- **PATTERN_SIZE**: Tuple (width, height) specifying internal corner count in the chessboard
- **SQUARE_SIZE**: Physical size of each chessboard square in millimeters  
- **IMAGES_FOLDER**: Directory path containing calibration images

```
PATTERN_SIZE = (11, 7)    # Internal corners (width, height)
SQUARE_SIZE = 30.0        # Square size in millimeters
IMAGES_FOLDER = "data"    # Calibration images folder
```
```
uv run code.py
```

**Output files:**
- **Undistorted images**: Saved in `results/` directory with detected corners overlaid
- **Console output**: Detailed calibration results including intrinsic matrix, distortion coefficients, and reprojection error
- **Parameter summary**: Focal lengths, principal point, and distortion characteristics

**Interpreting results:**
- **Reprojection error < 1.0 pixels**: Excellent calibration quality
- **Reprojection error 1.0-2.0 pixels**: Good calibration, suitable for most applications  
- **Reprojection error > 2.0 pixels**: Poor calibration, consider retaking images or checking pattern dimensions
