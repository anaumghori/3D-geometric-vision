import os
import glob
import cv2 as cv
from ultralytics import YOLO
from pathlib import Path
import numpy as np

from utils import (load_calibration, extract_camera_params, compute_disparity, disparity_to_depth,
    detect_objects, calculate_distances, annotate_distances, save_depth_visualization, save_disparity_visualization)

class DistanceMeasurement:
    """Pipeline for stereo vision-based distance measurement."""
    
    def __init__(self, model_path: str = "yolo11s.pt"):
        self.model = YOLO(model_path)
        self.setup_directories()
        

    def setup_directories(self):
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.depth_dir = self.results_dir / "depth_maps"
        self.disparity_dir = self.results_dir / "disparity_maps"
        self.depth_dir.mkdir(exist_ok=True)
        self.disparity_dir.mkdir(exist_ok=True)
        

    def process_frame(self, left_img: np.ndarray, right_img: np.ndarray, 
                     focal_length: float, baseline: float, 
                     save_visualizations: bool = False, frame_idx: int = 0):
        """
        Process a single stereo pair to estimate object distances.
        
        Args:
            left_img: Left camera image (BGR)
            right_img: Right camera image (BGR)
            focal_length: Camera focal length
            baseline: Stereo baseline distance
            save_visualizations: Whether to save depth/disparity maps
            frame_idx: Frame index for saving visualizations
            
        Returns:
            Annotated image with distance measurements
        """
        # Convert to grayscale for stereo matching
        left_gray = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
        right_gray = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)
        
        # Compute disparity
        disparity = compute_disparity(left_gray, right_gray)
        
        # Convert to depth
        depth_map = disparity_to_depth(disparity, focal_length, baseline)
        
        # Save visualizations for first few frames
        if save_visualizations:
            save_disparity_visualization(
                disparity, 
                self.disparity_dir / f"{frame_idx + 1}.png"
            )
            save_depth_visualization(
                depth_map, 
                self.depth_dir / f"{frame_idx + 1}.png"
            )
        
        # Detect objects
        annotated_img, boxes, classes, confidences = detect_objects(left_img, self.model)
        
        # Calculate distances
        if len(boxes) > 0:
            distances = calculate_distances(depth_map, boxes)
            # Add distance annotations
            annotated_img = annotate_distances(annotated_img, boxes, distances)
        
        return annotated_img
    

    def run(self, left_folder: str, right_folder: str, calib_file: str):
        """
        Run the complete pipeline on stereo image sequences.
        
        Args:
            left_folder: Path to left camera images
            right_folder: Path to right camera images
            calib_file: Path to calibration file
        """
        # Load calibration
        proj_left, proj_right = load_calibration(calib_file)
        
        if proj_left is None or proj_right is None:
            raise ValueError("Failed to load calibration parameters")
        
        # Extract camera parameters
        cam_matrix_left, trans_left = extract_camera_params(proj_left)
        cam_matrix_right, trans_right = extract_camera_params(proj_right)
        
        # Calculate stereo parameters
        focal_length = cam_matrix_left[0, 0]
        baseline = float(abs(trans_left[0] - trans_right[0]))
        
        print(f"Focal length: {focal_length:.2f} pixels")
        print(f"Baseline: {baseline:.2f} units")
        
        # Get image lists
        left_images = sorted(glob.glob(f"{left_folder}/*.png"))
        right_images = sorted(glob.glob(f"{right_folder}/*.png"))
        
        if len(left_images) != len(right_images):
            raise ValueError("Mismatch in number of left and right images")
        
        print(f"Found {len(left_images)} stereo pairs")
        
        # Setup video writer
        if len(left_images) > 0:
            # Read first image to get dimensions
            sample_img = cv.imread(left_images[0])
            height, width = sample_img.shape[:2]
            
            # Initialize video writer
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            video_path = self.results_dir / "output.mp4"
            video_writer = cv.VideoWriter(
                str(video_path), 
                fourcc, 
                10.0,  # 10 FPS
                (width, height)
            )
            
            if not video_writer.isOpened():
                raise ValueError("Failed to open video writer")
            
            print(f"Writing output video to: {video_path}")
        
        # Process each frame
        for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
            # Load images
            left_img = cv.imread(left_path)
            right_img = cv.imread(right_path)
            
            if left_img is None or right_img is None:
                print(f"Warning: Failed to load images at index {idx}")
                continue
            
            # Process frame (save visualizations only for first 3 frames)
            save_viz = idx < 3
            annotated_frame = self.process_frame(
                left_img, right_img, 
                focal_length, baseline,
                save_visualizations=save_viz,
                frame_idx=idx
            )
            
            # Write to video
            video_writer.write(annotated_frame)
            
            # Progress indicator
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(left_images)} frames")
        
        # Release video writer
        video_writer.release()
        print("Processing complete!")


if __name__ == "__main__":
    left_folder = "data/Set1"
    right_folder = "data/Set2"
    calibration_file = "data/calib_cam_to_cam.txt"
    pipeline = DistanceMeasurement()
    pipeline.run(left_folder, right_folder, calibration_file)