import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def load_calibration(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load camera projection matrices from KITTI calibration file.
    Args -> file_path: Path to calibration text file
    Returns -> Left and right camera projection matrices
    """
    proj_left = None
    proj_right = None
    
    with open(file_path, "r") as file:
        for line in file.readlines():
            if line.startswith("P_rect_02"):
                proj_left = np.array(line[10:].strip().split()).astype("float32").reshape(3, -1)
            elif line.startswith("P_rect_03"):
                proj_right = np.array(line[10:].strip().split()).astype("float32").reshape(3, -1)
    
    return proj_left, proj_right


def extract_camera_params(projection_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract camera intrinsics and translation from projection matrix.
    """
    cam_matrix, _, translation, _, _, _, _ = cv.decomposeProjectionMatrix(projection_matrix)
    # Convert from homogeneous coordinates
    translation = translation / translation[3]
    return cam_matrix, translation


def compute_disparity(left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
    """
    Calculate disparity map from stereo image pair.
    Args -> left_gray and light_gray: Left and Right grayscale images
    Returns -> Disparity map as float32 array
    """
    stereo = cv.StereoSGBM.create(
        minDisparity=0,
        numDisparities=96, # Must be a multiple of 16.
        blockSize=11,
        P1=8 * 3 * 36, # 8 * 3 * window_size^2
        P2=32 * 3 * 36, # 32 * 3 * window_size^2
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(left_gray, right_gray)
    # OpenCV returns disparity * 16, normalize it
    return disparity.astype(np.float32) / 16.0 


def disparity_to_depth(disparity: np.ndarray, focal_length: float, baseline: float) -> np.ndarray:
    """
    Convert disparity map to depth map using stereo geometry.

    Args:
        disparity: Disparity map
        focal_length: Camera focal length in pixels
        baseline: Distance between stereo cameras
        
    Returns -> Depth map in same units as baseline
    """
    disp_copy = disparity.copy()
    # Handle invalid disparity values
    disp_copy[disp_copy <= 0] = 0.1  # Prevent division by zero
    
    # Calculate depth: Z = (f * B) / d
    depth_map = (focal_length * baseline) / disp_copy
    return depth_map


def detect_objects(image: np.ndarray, model: YOLO) -> tuple[np.ndarray, list, list, list]:
    """
    Detect and track vehicles in the image.
    
    Args:
        image: Input BGR image
        model: YOLO model instance
        
    Returns -> Annotated image, bounding boxes, class names, confidence scores
    """

    # Track vehicles: bicycle(2), car(3), bus(6), truck(8)
    vehicle_classes = [2, 3, 6, 8]
    results = model.track(image, classes=vehicle_classes, persist=True)
    if len(results) == 0:
        return image, [], [], []
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return image, [], [], []
    
    # Get annotated image
    annotated = result.plot(conf=False, font_size=6.0)
    # Extract detection data
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = [result.names[int(cls)] for cls in result.boxes.cls]
    confidences = result.boxes.conf.cpu().numpy().tolist()
    
    return annotated, boxes, classes, confidences


def calculate_distances(depth_map: np.ndarray, boxes: list) -> list:
    """
    Calculate distance to each detected object using median depth.
    """
    distances = []
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        # Extract depth values within bounding box
        roi_depth = depth_map[y1:y2, x1:x2]
        if roi_depth.size > 0:
            # Use median for robust distance estimation
            distance = np.median(roi_depth)
        else:
            distance = 0
            
        distances.append(distance)
    
    return distances


def annotate_distances(image: np.ndarray, boxes: list, distances: list) -> np.ndarray:
    """
    Draw distance annotations on the image.
    
    Args:
        image: Input image (BGR)
        boxes: Bounding boxes
        distances: Distance values for each box
        
    Returns -> Annotated image
    """
    annotated = image.copy()
    
    for box, distance in zip(boxes, distances):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Format distance text
        text = f"{distance:.1f}m"
        
        # Calculate text size for background
        (text_width, text_height), _ = cv.getTextSize(
            text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw background rectangle
        cv.rectangle(
            annotated,
            (cx - text_width // 2 - 2, cy - text_height // 2 - 2),
            (cx + text_width // 2 + 2, cy + text_height // 2 + 2),
            (0, 180, 0),
            -1
        )
        
        # Draw text
        cv.putText(
            annotated,
            text,
            (cx - text_width // 2, cy + text_height // 2),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv.LINE_AA
        )
    
    return annotated


def save_depth_visualization(depth_map: np.ndarray, filename: str):
    """Save depth map as a color-mapped image."""
    plt.figure(figsize=(12, 8))
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar(label='Distance (m)')
    plt.title('Depth Map')
    plt.axis('off')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def save_disparity_visualization(disparity: np.ndarray, filename: str):
    """Save disparity map as a color-mapped image."""
    plt.figure(figsize=(12, 8))
    plt.imshow(disparity, cmap='viridis')
    plt.colorbar(label='Disparity (pixels)')
    plt.title('Disparity Map')
    plt.axis('off')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()