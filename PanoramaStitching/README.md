# Panorama Stitching 

### Table of Contents
- [What is Panorama Stitching?](#what-is-panorama-stitching)
- [Panorama Stitching Steps](#panorama-stitching-steps)
  - [Step 1: Feature Detection](#step-1-feature-detection)
  - [Step 2: Feature Description](#step-2-feature-description)
  - [Step 3: Feature Matching](#step-3-feature-matching)
  - [Step 4: Homography Estimation](#step-4-homography-estimation)
  - [Step 5: Image Warping](#step-5-image-warping)
  - [Step 6: Canvas Size and Blending](#step-6-canvas-size-and-blending)
- [Mathematical Foundations](#mathematical-foundations)
  - [Homography Matrix](#homography-matrix)
  - [Direct Linear Transform (DLT)](#direct-linear-transform-dlt)
  - [Image Warping Transformation](#image-warping-transformation)
  - [RANSAC for Robust Estimation](#ransac-for-robust-estimation)
- [Implementation Overview](#implementation-overview)
  - [Usage](#usage)



<br><br>


# What is Panorama Stitching?

Panorama stitching is a computer vision technique that combines multiple overlapping images to create a single, wide-field image called a panorama. This process mimics what our eyes naturally do when we look around a scene; it reconstructs the broader view by aligning and blending multiple perspective views.  

At its core, panorama stitching solves a geometric puzzle: "How do I align multiple images of the same scene taken from different positions?"
The solution involves finding the mathematical transformation that maps pixels from one image to their corresponding positions in another image. This transformation accounts for:  
- Camera rotation between shots
- Perspective changes due to different viewpoints
- Lens distortion effects
- Scale variations from zoom differences

The fundamental challenge is that each image captures the scene from a slightly different viewpoint, with different lighting conditions, and potential camera movement between shots. The algorithm must intelligently find matching features between adjacent images, calculate how to geometrically align them and blend them seamlessly to hide the seams. 

### Mathematical Prerequisites
In panorama stitching, we use **homogeneous coordinates** to represent 2D points. A point $(x, y)$ is expressed as a 3D vector $[x, y, 1]$, which allows us to handle **translations, rotations, scaling, and perspective transformations** uniformly through matrix multiplication. Homogeneous coordinates also enable the representation of **points at infinity**, which is particularly useful for handling parallel lines that appear to meet at infinity. 

The underlying mathematics is based on projective geometry, which describes how 3D points are projected onto 2D image planes. A key insight is that when a camera rotates or changes viewpoint, the relationship between corresponding points in different images follows a homography (a projective transformation). The detailed mathematical formulations and equations for these concepts are explained in the [Mathematical Foundations](#mathematical-foundations) section.

<br><br>

| Input 1 | Input 2 | Input 3 | Result |
|---------|---------|---------|--------|
| ![Input 1](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/data/Set3/1.jpg) | ![Input 2](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/data/Set3/2.jpg) | ![Input 3](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/data/Set3/3.jpg) | ![Result](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/data/Set3/Final_Panorama.jpg) |


<br><br>


# Panorama Stitching Steps

### Step 1: Feature Detection
Feature detection is the process of finding unique points in an image (like corners or edges) that stand out and can be recognized in other pictures of the same scene. In panorama stitching, these points are crucial because they act as anchors that help the algorithm figure out how images overlap and fit together. By matching these distinctive points across different views, the computer can establish correspondences and accurately align the images. To make this possible, feature detection relies on a few following important properties: 
1. **Corner detection** identifies points where image intensity changes sharply in multiple directions, making them easy to recognize. 
2. **Scale invariance** ensures that features can still be detected even if the same scene is captured from different distances. 
3. **Rotation invariance** allows features to remain recognizable regardless of how the camera is oriented.  

Common Algorithms:
- **SIFT (Scale-Invariant Feature Transform):** Detects blob-like features that are stable across scale and rotation changes
- **SURF (Speeded-Up Robust Features):** Faster alternative to SIFT with similar properties
- **ORB (Oriented FAST and Rotated BRIEF):** Efficient algorithm suitable for real-time applications

<br><br>

### Step 2: Feature Description
Once feature points are detected, the algorithm creates a unique "fingerprint," called a descriptor, for each one. These descriptors capture the local appearance around a feature in a way that remains reliable even under changes in lighting or slight variations in perspective. They are essential because simply finding points isn't enough, the algorithm also needs a way to describe what those points look like so they can be correctly matched between images. An analogy is describing a person's face: you might note a prominent nose, closely set eyes, or a wide forehead. In the same way, feature descriptors record the distinctive characteristics of image patches, making each feature point recognizable across different views.

How it works:
1. **Local Gradients:** Measure how pixel brightness changes in direction and strength around each feature point.
2. **Histogram Creation:** Summarize these gradient directions into histograms, which capture the dominant patterns in the neighborhood.
3. **Normalization:** Adjust the descriptor values so they are less sensitive to lighting or contrast changes, making matches more reliable.

<br><br>

### Step 3: Feature Matching
Feature matching is the process of finding corresponding feature points between overlapping images, which is essential for aligning them accurately. The challenge is that not every detected feature is a true match; similar looking but unrelated points, repetitive patterns like windows or tiles, and lighting differences can all lead to false matches. The goal is to reliably pair features from one image with their counterparts in another so the algorithm knows how the images overlap.

How it works:
1. **Distance Calculation:** Compare feature descriptors using similarity measures such as Euclidean distance.
2. **Nearest Neighbor Search:** For each feature in image A, find the most similar feature in image B.
3. **Ratio Test:** Check that the best match is significantly better than the second-best to reduce ambiguity.

<br><br>

| Example 1 | Example 2 |
|----------|----------|
| ![Example 1](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/readme_images/feature_matches1.jpg)         | ![Example 2](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/readme_images/feature_matches2.jpg)         |

<br><br>

### Step 4: Homography Estimation
Homography estimation calculates the geometric transformation that best aligns the matched feature points between two images. This step is crucial because, once we know which points correspond, we need a mathematical function that maps coordinates from one image to the other. A homography can model several types of transformations: simple shifts (translation), camera rotations, changes in zoom (scale), tilts and viewpoint changes (perspective), and even skewing effects (shear).

Estimation process:
1. **Direct Linear Transform (DLT):** Form equations using the matched point pairs.
2. **Least Squares Solution:** Solve for the homography parameters that minimize the reprojection error.
3. **Minimum Requirements:** At least 4 point correspondences are needed to compute 8 unknown parameters.

However, not all feature matches are correct, and outliers can distort the result. To address this, **RANSAC (Random Sample Consensus)** is used to robustly estimate the homography by identifying and excluding mismatched points. RANSAC is like fitting a line through noisy data points: most points follow the trend, but a few outliers don't. By testing small random subsets repeatedly, RANSAC finds the line (or homography) that most points agree with.

How RANSAC Works:
1. **Random Sampling:** Randomly select a minimal set of matches (4 points for homography)
2. **Model Fitting:** Calculate a homography using only these points
3. **Consensus Testing:** Count how many other matches agree with this homography (within a threshold)
4. **Iteration:** Repeat many times, keeping the homography with the most consensus
5. **Refinement:** Recalculate the final homography using all inlier matches

<br><br>

### Step 5: Image Warping
Image warping transforms one image according to the calculated homography so it aligns with another image. This is necessary because, once we know the geometric relationship, we must actually reshape the pixels to match the perspective. 

In practice, there are two main strategies: **(1) forward mapping**, where each source pixel is projected into the destination, and **(2) inverse mapping**, where each destination pixel is traced back to its source. Panorama stitching almost always uses inverse mapping, since it avoids gaps (holes) in the output image. Because mapped coordinates often fall between pixels, interpolation methods like **(3) bilinear interpolation** are applied to estimate the correct intensity values.

Challenges:
1. **Sampling:** Source and destination pixels don't always align perfectly on the grid.
2. **Holes:** Some destination pixels may not map to any source pixel.
3. **Multiple Sources:** A single destination pixel may get contributions from several source pixels.

<br><br>

| Example 1 | Example 2 |
|----------|----------|
| ![Example 1](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/readme_images/warped1.jpg)         | ![Example 2](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/readme_images/warped2.jpg)         |

<br><br>

### Step 6: Canvas Size and Blending
Once images are warped, the algorithm must first determine how large the output canvas needs to be and then seamlessly merge overlapping regions. Warping often pushes image content beyond its original boundaries, so the canvas must be expanded to fit everything. Afterward, blending is applied to smooth seams caused by exposure differences, small misalignments, or parallax effects, ensuring the panorama looks natural.

There are several blending challenges: Differences in exposure may cause variations in brightness or color tones across images, making the seams noticeable. Even when exposures match, overlaps can create visible edges if not blended smoothly. Additionally, parallax effects, caused by objects at different depths, can lead to misalignments that are difficult to correct with a single homography, further complicating the blending process.

<br><br>

| Final Panorama | Final Panorama |
|----------|----------|
| ![Final image](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/data/Set2/Final_Panorama.jpg)         | ![Final image](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/data/Set1/Final_Panorama.jpg)         |


<br><br>


# Mathematical Foundations

In panorama stitching, 2D image points are expressed in homogeneous coordinates, where a Cartesian point $(x, y)$ is written as $[x, y, 1]^T$. Here, the superscript "T" denotes transpose, meaning the point is represented as a vertical column to enable proper matrix multiplication. This representation is a clever mathematical trick that simplifies handling geometric transformations, since operations such as translation, rotation, scaling, and tilting can all be performed using a single matrix multiplication instead of requiring separate procedures for each transformation.

**What are point correspondences?**
Point correspondences are like matching pairs of landmarks between two photos. Imagine you take two overlapping photos of a building. A window corner that appears in both photos represents the same physical point in the real world, but it has different pixel coordinates in each image. These matching points are our correspondences.

When we have corresponding points between two overlapping images, we establish point correspondences:

$\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \mathbf{p_1} = [x_1, y_1, 1]^T \leftrightarrow \mathbf{p_1'} = [x_1', y_1', 1]^T$

$\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \mathbf{p_2} = [x_2, y_2, 1]^T \leftrightarrow \mathbf{p_2'} = [x_2', y_2', 1]^T$

$\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots$

$\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \mathbf{p_n} = [x_n, y_n, 1]^T \leftrightarrow \mathbf{p_n'} = [x_n', y_n', 1]^T$

**Breaking down this notation:**
- $\mathbf{p_1}$: The first matching point in image 1
- $\mathbf{p_1'}$: The same physical point as it appears in image 2 (the prime mark ' indicates "second image")
- $\leftrightarrow$: This symbol means "corresponds to" or "matches with"
- The subscripts (1, 2, ..., n) just number our matching point pairs

These correspondences represent the same physical points in the scene as observed from two different camera positions.

### Homography Matrix
A homography matrix is like a mathematical recipe that transforms coordinates from one image to another. Since we represent points in homogeneous coordinates as [x,y,1], this requires a 3×3 matrix where each element has a specific role in controlling transformation. The relationship between corresponding points in two images is expressed as:

$$\begin{bmatrix} x' \\\ y' \\\ w' \end{bmatrix} 
= \begin{bmatrix} 
h_{11} & h_{12} & h_{13} \\\ 
h_{21} & h_{22} & h_{23} \\\ 
h_{31} & h_{32} & h_{33} 
\end{bmatrix} 
\begin{bmatrix} x \\\ y \\\ 1 \end{bmatrix}$$

**Understanding each part of the homography matrix:**
- **Top-left 2×2 block** $(h_{11}, h_{12}; h_{21}, h_{22})$: Controls rotation, scaling, and shearing
- **Top-right column** $(h_{13}; h_{23})$: Controls translation (shifting left/right and up/down)
- **Bottom row** $$([h_{31}, h_{32}, h_{33}])$$: Controls perspective effects (making things look closer or farther)

**What happens in this multiplication?**
When we multiply the matrix with our point $[x, y, 1]$, we get:
- $x' = h_{11}x + h_{12}y + h_{13} \times 1$
- $y' = h_{21}x + h_{22}y + h_{23} \times 1$  
- $w' = h_{31}x + h_{32}y + h_{33} \times 1$

The final 2D coordinates are obtained by normalizing: $\quad \quad x_{final} = \frac{x'}{w'}, \quad \quad y_{final} = \frac{y'}{w'}$

**Why do we divide by w'?**
This is the key to homogeneous coordinates! The third coordinate $w'$ acts like a "zoom factor." When $w' = 1$, we get regular coordinates. When $w' > 1$, it's like the point is closer (coordinates shrink after division). When $w' < 1$, the point is farther away (coordinates expand). This normalization converts homogeneous coordinates back into regular 2D pixel coordinates. Or more compactly: $\mathbf{p'} = \mathbf{H} \mathbf{p}$.

A homography has 8 degrees of freedom, since the 9 matrix values can be scaled by setting $h_{33} = 1$, leaving only 8 unique parameters. To solve for these, we need at least 4 point correspondences: each pair of matching points provides two equations (one for $x$ and one for $y$), so 4 pairs give the 8 equations needed. This transformation works under the assumption that the observed scene is approximately planar or that the camera undergoes pure rotation around its optical center, conditions under which the mapping between images can be accurately modeled by a single homography matrix.

### Direct Linear Transform (DLT)

DLT is our method for finding the mystery homography matrix $\mathbf{H}$. We know the input points and output points, but we need to figure out the transformation matrix that connects them. It's like reverse-engineering a recipe when you know the ingredients and the final dish.

The Direct Linear Transform (DLT) is the standard method for computing the homography matrix $\mathbf{H}$. We know the input points and output points, but we need to figure out the transformation matrix that connects them. Instead of guessing the transformation, we "reverse-engineer" it by enforcing the condition that a point $\mathbf{p_i}$ in one image, after transformation, should align with its match $\mathbf{p_i'}$ in the other. Mathematically, this is expressed as:

$\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \mathbf{p_i'} \times (\mathbf{H} \mathbf{p_i}) = \mathbf{0}$

The cross product is used here because homogeneous coordinates allow scaling; two points like $[2,4,2]$ and $[1,2,1]$ represent the same Cartesian location. The cross product ignores such scaling and ensures the transformed point $\mathbf{H}\mathbf{p_i}$ is parallel to the target $\mathbf{p_i'}$. Expanding this constraint gives two linear equations per correspondence:

$\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad x_i' (h_{31} x_i + h_{32} y_i + h_{33}) - (h_{11} x_i + h_{12} y_i + h_{13}) = 0$

$\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad y_i' (h_{31} x_i + h_{32} y_i + h_{33}) - (h_{21} x_i + h_{22} y_i + h_{23}) = 0$

These look complex, but they're just rearranging our homography multiplication! 
- The first equation comes from setting the x-components equal
- The second equation comes from setting the y-components equal  
- Each equation is linear in the unknown h-values (no squares or other powers)

By stacking these equations from multiple correspondences, we form a linear system $\mathbf{A} \mathbf{h} = \mathbf{0}$, where $\mathbf{h}$ is the vectorized version of the homography matrix, $\mathbf{h} = [h_{11}, h_{12}, h_{13}, h_{21}, h_{22}, h_{23}, h_{31}, h_{32}, h_{33}]^T$. The matrix $\mathbf{A}$ contains the coefficients of these equations, with size $2n \times 9$ for $n$ correspondences. Solving this system requires finding a non-trivial $\mathbf{h}$ that satisfies the constraints.

This is done using Singular Value Decomposition (SVD), which provides the solution that minimizes $||\mathbf{A} \mathbf{h}||^2$ under the condition $||\mathbf{h}|| = 1$. This prevents the trivial zero solution and finds the best fit for $\mathbf{H}$. When more than the minimum number of correspondences is available, the system becomes overdetermined, and SVD naturally handles this by balancing the equations, making the solution more robust to noise and measurement errors.

### Image Warping Transformation

**What is image warping?**
Image warping is the process of reshaping one image so it aligns with another, similar to stretching and bending a rubber sheet until it matches a new perspective. Once we have the homography matrix, we can warp one image onto another by mapping each pixel in the target image $(x', y')$ back to its corresponding location in the source image using the inverse transformation:

$\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad [x, y, w]^T = \mathbf{H}^{-1} [x', y', 1]^T$

We use the inverse homography because this backward mapping ensures that every pixel in the destination image knows where to fetch its value from in the source. In contrast, forward mapping sends source pixels to new positions, which can leave gaps (empty pixels) or overlaps (multiple pixels landing in the same spot). You can think of forward mapping like a crowd of people (source pixels) rushing to claim seats (destination pixels), some seats stay empty while others get overcrowded. Backward mapping, on the other hand, is like every seat (destination pixel) calling out for the right person to sit there, ensuring that every position is filled exactly once. After applying $\mathbf{H}^{-1}$, the source coordinates are then normalized to obtain:

$\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad x_{source} = \frac{x}{w}, \quad y_{source} = \frac{y}{w}$  

These coordinates are rarely integers because transformations often place pixels at fractional positions (e.g., $(10, 20)$ may map to $(10.3, 20.7)$). Since image pixels only exist at integer coordinates, we estimate the pixel intensity at these non-integer locations using bilinear interpolation. This method considers the four nearest neighboring pixels ($(10,20)$, $(11,20)$, $(10,21)$, and $(11,21)$ in this example) and computes a weighted average, where closer pixels contribute more strongly. Conceptually, it's like mixing paint colors in proportion to how near each neighbor is, producing a smooth and realistic warped image.

### RANSAC for Robust Estimation

Even with strong feature detection, some matches between images will be incorrect (outliers), and if used directly, they can completely distort the estimated homography. To address this, we use RANSAC (Random Sample Consensus), which acts like a quality control system by repeatedly testing small random subsets of matches and discarding bad ones. For each candidate homography $\mathbf{H}_k$, the algorithm measures the reprojection error for every correspondence:

$\quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad e_i = \left\| \frac{\mathbf{H}_k \mathbf{p_i}}{(\mathbf{H}_k \mathbf{p_i})_3} - \mathbf{p_i'} \right\|_2$

**Breaking down this error formula:**

1. **$\mathbf{H}_k \mathbf{p_i}$**: Apply our candidate homography to point $\mathbf{p_i}$  

2. **$\frac{\mathbf{H}_k \mathbf{p_i}}{(\mathbf{H}_k \mathbf{p_i})_3}$**: Normalize the homogeneous coordinates (divide by the third component)  

3. **$- \mathbf{p_i'}$**: Subtract where the point should actually be  

4. **$\left\| ... \right\|_2$**: Calculate the Euclidean distance (how far off we are)  


If the error $e_i$ is below a threshold $\tau$ (often 1–3 pixels), the match is considered an inlier; otherwise, it is an outlier. A perfect homography would make all $e_i$ values zero, but in practice, small errors are tolerated due to noise.

RANSAC iteratively repeats this process: it selects four random correspondences to compute a temporary homography, evaluates all matches using the reprojection error, and counts the number of inliers. The homography with the largest consensus set of inliers is chosen as the best estimate. Finally, the homography is recomputed using all inliers through least squares refinement, yielding a more accurate and stable result. In essence, instead of letting a few bad matches dictate the outcome, RANSAC lets all correspondences "vote," and the transformation supported by the majority wins.  


<br><br>

# Implementation Overview
This project implements a hierarchical panorama stitching pipeline using Python and OpenCV, balancing computational efficiency with high-quality output. Instead of stitching images sequentially, the pipeline uses a **hierarchical stitching strategy**, which addresses the limitations of sequential stitching.

In sequential stitching, images are combined one by one. While straightforward, this method accumulates small alignment errors at each step, resulting in quality degradation and geometric distortions in the final panorama. 

The hierarchical approach stitches images in stages, pairing adjacent images first, then combining the resulting intermediate panoramas, and continuing iteratively until the final panorama is formed. Following is an example with 6 images [A, B, C, D, E, F]:   
Level 1: ``A + B → AB``,  ``C + D → CD``, ``E + F → EF`` (3 results)  
Level 2: ``AB + CD → ABCD``, ``EF waits`` (2 results)  
Level 3: ``ABCD + EF → Final Panorama``  

This project does not create custom implementations for the core computer vision operations. Instead, it leverages OpenCV's optimized functions for feature detection, feature matching, homography estimation, image warping, and distance transform calculations. OpenCV provides battle-tested implementations of these complex algorithms that are both faster and more robust than custom implementations would be.

### Usage
Parameters:  
-- InputPath (required): Directory containing input images (JPG format)  

```
uv run code.py --InputPath /path/to/images/
```

All test images for experimenting with the panorama stitching algorithm can be found in the `data` directory. The processed results, including the final panorama and any intermediate outputs, will also be saved in the same `data` directory.

During the stitching process, a `temp_pano` folder will be automatically created within each image set's directory. This temporary folder contains:  
- **Feature matching visualizations** (`matches_pair0.jpg`, `matches_pair1.jpg`, etc.) - showing detected keypoints and their correspondences between adjacent images
- **Intermediate stitching results** (`0.jpg`, `1.jpg`, etc.) - partial panoramas generated during multi-image processing iterations
- **Multi-iteration matching images** (`matches_iter0_pair0.jpg`, etc.) - feature matches for each iteration when processing multiple images

**Note:** This implementation prioritizes educational clarity and practical usability. For production applications requiring the highest quality results, consider implementing additional techniques like bundle adjustment, cylindrical projection for wide-angle panoramas, or multi-band blending for seamless photometric alignment.
