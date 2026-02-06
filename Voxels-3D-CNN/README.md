# 3D Point Cloud Classification with Sparse Voxel CNNs

3D point cloud classification is a fundamental task in computer vision that involves categorizing three-dimensional objects represented as collections of points in space. Unlike traditional 2D images that capture appearance from a single viewpoint, point clouds represent the actual 3D geometry of objects, making them invaluable for applications like autonomous driving, robotics, and augmented reality.

This project implements a deep learning pipeline that classifies 3D objects from the ShapeNet dataset using sparse voxel convolutions. The approach transforms unstructured point clouds into a structured voxel grid representation, enabling efficient 3D convolutions while maintaining spatial sparsity.

### Point Clouds
A point cloud is a set of data points in 3D space, where each point is defined by its (x, y, z) coordinates. Point clouds are typically captured using 3D sensors like LiDAR or depth cameras, or generated from multiple 2D images. Unlike traditional mesh representations, point clouds are:
- **Unordered**: No inherent sequential structure
- **Irregular**: Non-uniform spacing between points
- **Sparse**: Most of 3D space is empty

### Voxels
Voxels (volumetric pixels) are the 3D equivalent of 2D pixels. Just as an image divides a 2D plane into a grid of pixels, voxelization divides 3D space into a regular grid of cubic cells. Each voxel can store information about the points it contains, such as:
- **Occupancy**: Whether the voxel contains any points
- **Features**: Aggregated properties of points within the voxel (mean position, color, normal vectors)
- **Point count**: Number of points falling within the voxel

Voxelization transforms irregular point clouds into a structured representation that standard 3D CNNs can process. However, since most voxels are empty, **sparse convolutions** are used to skip computation on empty regions, making the approach both memory and computationally efficient.

<br></br>

| Point Cloud | Voxels |
|----------|----------|
| <img width="298" height="332" alt="chair_point_cloud" src="https://github.com/user-attachments/assets/bb8f43cf-a2b1-4340-935d-43d23aa580e4" />         | <img width="266" height="284" alt="chair_voxels" src="https://github.com/user-attachments/assets/09704852-4408-4424-86ea-cf5005c46eaa" />         |

<br></br>

## ⚠️ Important Setup Requirements

This code may not work out of the box due to the following environment-specific requirements:

1. **Python Version Constraint**: The `spconv` library requires Python 3.11 or lower to function properly. During development, the library caused issues even in a Python 3.11 environment, requiring a downgrade to **Python 3.10** for stable operation.

2. **CUDA Requirement**: On Windows, spconv requires a CUDA-enabled NVIDIA GPU and the CUDA Toolkit installed on the system. On Linux, users may install either the GPU build (requires only an NVIDIA driver) or a CPU-only build, though CPU performance is significantly slower. Correctly matching the installed spconv wheel to your CUDA setup is essential to avoid runtime or import errors. Refer to the official documentation: [spconv installation guide](https://github.com/traveller59/spconv).

3. **Isolated Environment Recommended**: That being said, the `pyproject.toml` and other files in the main project folder will not work for this specific project due to conflicting dependency requirements. This project was developed in a separate Python environment on an RTX 4000 Ada GPU rented via RunPod. You will need to create an isolated environment specifically for this project.

<br></br>

## Installation

#### 1. Create Python 3.10 Environment
Create a new Python 3.10 environment using `uv`:
```
uv init --python 3.10
```

#### 2. Install Dependencies
For library installation, refer to the [spconv documentation](https://github.com/traveller59/spconv) to install the correct version compatible with your CUDA installation. After installing spconv, install the remaining required libraries:

```
uv add torch torchvision
uv add numpy pandas matplotlib plotly scikit-learn tqdm cumm
```

**Note**: Ensure PyTorch is installed with CUDA support matching your system's CUDA version.

#### 3. Download Dataset
Download the **ShapeNet Core** dataset from the official source: [ShapeNet.org](https://shapenet.org/). After downloading, place the dataset in a folder named `shapenet-core-dataset` in the project directory:
```
Voxels-3D-CNN/
├── shapenet-core-dataset/
│   ├── train_split.json
│   ├── val_split.json
│   ├── test_split.json
│   └── [point cloud files]
├── main.py
├── model.py
├── training.py
└── visualization.py
```

#### 4. Run the training script
```
uv run main.py
```

<br></br>

## Training Process
- The model trains for **3 epochs** on the ShapeNet dataset
- Training progress is displayed via tqdm progress bars
- The best model checkpoint is saved based on validation loss
  
All outputs are saved in the `results/` directory:

1. **Model Checkpoint**: `PointCloudClassifier.pt` - The trained model weights
2. **Training Curves**: `training_curves.png` - Matplotlib visualization showing training and validation loss over epochs
3. **Prediction Visualization**: `predictions.html` - An **interactive HTML visualization** featuring:
   - **Column 1**: Original point cloud rendered as 3D scatter plot
   - **Column 2**: Voxelized representation showing occupied voxels as 3D grid
   - **Column 3**: Prediction results displaying the true label vs. predicted label with color-coded accuracy (green for correct, red for incorrect)

The visualization uses Plotly for interactive 3D exploration, allowing you to rotate, zoom, and inspect the point clouds and voxel grids from any angle. 

<br></br>


## Architecture Overview
The pipeline consists of four main components:

1. **Voxelization** (`GenerateVoxel`): Converts raw point clouds into voxel grids using the Point2Voxel algorithm
2. **Voxel Encoding** (`MeanVoxelEncoder`): Aggregates point features within each voxel by averaging
3. **Sparse 3D CNN Backbone** (`SparseCNNBackbone`): Extracts hierarchical features using sparse convolutions
4. **Classification Head** (`ClassificationHead`): Maps encoded features to class probabilities

The model is trained end-to-end using cross-entropy loss and the Adam optimizer with a OneCycleLR learning rate scheduler for faster convergence.