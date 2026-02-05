# Camera Calibration Using Zhang's Method

Camera calibration is the process of determining the internal geometric and optical characteristics of a camera system. Think of it as creating a precise mathematical model that describes how your camera transforms the 3D world into 2D images. Just as you might calibrate a scale to ensure accurate measurements, camera calibration ensures that your camera produces geometrically accurate representations of the real world.

The fundamental question camera calibration answers is: **"Given a 3D point in the world, where will it appear in my camera's image, and vice versa?"** This relationship is crucial for applications like: 3D reconstruction, Augmented reality, Robot vision etc.  

Camera calibration determines two types of parameters:
1. **Intrinsic parameters**: Properties internal to the camera (focal length, optical center, distortion)
2. **Extrinsic parameters**: The camera's position and orientation in 3D space relative to the calibration pattern

Real cameras deviate from ideal mathematical models in several ways:
- **Lens distortion**: Straight lines in the world appear curved in images, especially near image borders
- **Manufacturing variations**: No two cameras are identical, even from the same model
- **Optical imperfections**: Lenses introduce various aberrations and distortions
- **Unknown focal length**: The effective focal length varies with focus settings and manufacturing tolerances

Without calibration, measurements from images would be inaccurate, 3D reconstruction would fail, and computer vision algorithms would produce unreliable results.

### The Pinhole Camera Model

The pinhole camera model is the fundamental mathematical framework for understanding how 3D points project onto 2D images. Imagine a box with a tiny hole in one side, light rays from the outside world pass through this hole and create an inverted image on the opposite wall. This simple concept forms the basis for all camera calibration mathematics. 

In the pinhole model, every 3D point in the world projects through a single point (the camera center) onto the image plane. This creates the perspective effect we see in photographs; objects farther away appear smaller, and parallel lines converge to vanishing points. Every visible point in the 3D world reflects light in all directions, but the tiny pinhole acts as a selective filter, allowing exactly one ray from each point to pass through. All other rays are blocked by the camera walls, ensuring that each world point contributes to exactly one pixel location. This process happens simultaneously for all visible points, with millions of rays passing through the same tiny opening at once, each following its own geometric path to create a complete, sharp image. The resulting image is both horizontally and vertically inverted because rays from the top of objects travel downward through the pinhole to hit the bottom of the image plane, while rays from the left side of objects end up on the right side of the image, creating the characteristic flipped appearance that our brains (or camera software) must correct.

#### Key Assumptions of the Pinhole Model

1. **Single projection center**: All light rays pass through one point
2. **Linear perspective**: Straight lines remain straight (before distortion)
3. **No lens effects**: The ideal model ignores lens distortion, depth of field, and other optical effects
4. **Instantaneous capture**: No motion blur or temporal effects

<br><br>

| Calibration Result 1 | Calibration Result 2 |
|----------|----------|
| ![Result 1](https://github.com/anaumghori/3D-geometric-vision/blob/main/CameraCalibration/results/undistorted_3.png) | ![Result 2](https://github.com/anaumghori/3D-geometric-vision/blob/main/CameraCalibration/results/undistorted_20.png) |

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

**Step 1: Download the calibration dataset**

Download the calibration images from [Kaggle - Stereo Camera Chessboard Pictures](https://www.kaggle.com/datasets/danielwe14/stereocamera-chessboard-pictures) and place them in the `data` folder.

**Step 2: Run the calibration**

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
