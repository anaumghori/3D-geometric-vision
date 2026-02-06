# Panorama Stitching 

Panorama stitching is a computer vision technique that combines multiple overlapping images to create a single, wide-field image called a panorama. This process mimics what our eyes naturally do when we look around a scene; it reconstructs the broader view by aligning and blending multiple perspective views.  

At its core, panorama stitching solves a geometric puzzle: "How do I align multiple images of the same scene taken from different positions?"
The solution involves finding the mathematical transformation that maps pixels from one image to their corresponding positions in another image. This transformation accounts for:  
- Camera rotation between shots
- Perspective changes due to different viewpoints
- Lens distortion effects
- Scale variations from zoom differences

The fundamental challenge is that each image captures the scene from a slightly different viewpoint, with different lighting conditions, and potential camera movement between shots. The algorithm must intelligently find matching features between adjacent images, calculate how to geometrically align them and blend them seamlessly to hide the seams. 

In panorama stitching, we use **homogeneous coordinates** to represent 2D points. A point $(x, y)$ is expressed as a 3D vector $[x, y, 1]$, which allows us to handle **translations, rotations, scaling, and perspective transformations** uniformly through matrix multiplication. Homogeneous coordinates also enable the representation of **points at infinity**, which is particularly useful for handling parallel lines that appear to meet at infinity. 

<br><br>

| Input 1 | Input 2 | Input 3 | Result |
|---------|---------|---------|--------|
| ![Input 1](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/data/Set3/1.jpg) | ![Input 2](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/data/Set3/2.jpg) | ![Input 3](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/data/Set3/3.jpg) | ![Result](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/data/Set3/Final_Panorama.jpg) |


<br><br>


### Feature Matching
Feature matching is the process of finding corresponding feature points between overlapping images, which is essential for aligning them accurately. The challenge is that not every detected feature is a true match; similar looking but unrelated points, repetitive patterns like windows or tiles, and lighting differences can all lead to false matches. The goal is to reliably pair features from one image with their counterparts in another so the algorithm knows how the images overlap.

How it works:
1. **Distance Calculation:** Compare feature descriptors using similarity measures such as Euclidean distance.
2. **Nearest Neighbor Search:** For each feature in image A, find the most similar feature in image B.
3. **Ratio Test:** Check that the best match is significantly better than the second-best to reduce ambiguity.

| Example 1 | Example 2 |
|----------|----------|
| ![Example 1](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/readme_images/feature_matches1.jpg)         | ![Example 2](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/readme_images/feature_matches2.jpg)         |

<br><br>

### Image Warping
Image warping transforms one image according to the calculated homography so it aligns with another image. This is necessary because, once we know the geometric relationship, we must actually reshape the pixels to match the perspective. 

In practice, there are two main strategies: **(1) forward mapping**, where each source pixel is projected into the destination, and **(2) inverse mapping**, where each destination pixel is traced back to its source. Panorama stitching almost always uses inverse mapping, since it avoids gaps (holes) in the output image. Because mapped coordinates often fall between pixels, interpolation methods like **(3) bilinear interpolation** are applied to estimate the correct intensity values.

Challenges:
1. **Sampling:** Source and destination pixels don't always align perfectly on the grid.
2. **Holes:** Some destination pixels may not map to any source pixel.
3. **Multiple Sources:** A single destination pixel may get contributions from several source pixels.

| Example 1 | Example 2 |
|----------|----------|
| ![Example 1](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/readme_images/warped1.jpg)         | ![Example 2](https://github.com/anaumghori/3D-geometric-vision/blob/main/PanoramaStitching/readme_images/warped2.jpg)         |


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