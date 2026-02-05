import numpy as np
import cv2
import glob
import os
from scipy.optimize import least_squares

class CameraCalibrator:
    def __init__(self, pattern_size, square_size_mm, images_folder):
        """
        pattern_size: Tuple (width, height) of internal corners in chessboard
        square_size_mm: Physical size of each square in millimeters
        images_folder: Path to folder containing calibration images
        """
        self.pattern_size = pattern_size
        self.square_size = square_size_mm
        self.images_folder = images_folder
        self.object_points = []
        self.image_points = []
        self.homographies = []
        
    def detect_corners(self, image_path):
        """Detect chessboard corners and return world/image point pairs."""
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        
        if found:
            # Refine corner positions to subpixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            world_pts = self._create_world_points()
            return world_pts, refined_corners.reshape(-1, 2)
        
        return None, None
    
    def _create_world_points(self):
        """Create 3D world coordinates for the chessboard pattern."""
        points = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        points[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        return points * self.square_size
    
    def _build_constraint_matrix(self):
        """Build the constraint matrix V from homographies for Zhang's method."""
        def v_ij(H, i, j):
            """Create constraint vector v_ij from homography matrix."""
            return np.array([
                H[0, i] * H[0, j],
                H[0, i] * H[1, j] + H[1, i] * H[0, j],
                H[1, i] * H[1, j],
                H[2, i] * H[0, j] + H[0, i] * H[2, j],
                H[2, i] * H[1, j] + H[1, i] * H[2, j],
                H[2, i] * H[2, j]
            ])
        
        V = []
        for H in self.homographies:
            # Two constraints per homography: h1^T * B * h2 = 0 and h1^T * B * h1 = h2^T * B * h2
            V.append(v_ij(H, 0, 1))  # h1^T * B * h2 = 0
            V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))  # h1^T * B * h1 - h2^T * B * h2 = 0
        
        return np.array(V)
    
    def _solve_for_intrinsics(self, V):
        """Solve for intrinsic parameters from constraint matrix."""
        # Solve Vb = 0 using SVD
        _, _, Vt = np.linalg.svd(V)
        b = Vt[-1, :]
        B11, B12, B22, B13, B23, B33 = b
        
        # Extract camera parameters from the conic matrix B
        v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
        lambda_val = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
        alpha = np.sqrt(lambda_val / B11)
        beta = np.sqrt(lambda_val * B11 / (B11 * B22 - B12**2))
        gamma = -B12 * alpha**2 * beta / lambda_val
        u0 = gamma * v0 / beta - B13 * alpha**2 / lambda_val
        
        return np.array([
            [alpha, gamma, u0],
            [0, beta, v0],
            [0, 0, 1]
        ])
    
    def extract_intrinsic_matrix(self):
        """Extract camera intrinsic parameters using Zhang's method."""
        V = self._build_constraint_matrix()
        return self._solve_for_intrinsics(V)
    
    def calculate_extrinsics(self, K):
        """Compute extrinsic parameters from homographies and intrinsics."""
        K_inv = np.linalg.inv(K)
        extrinsics = []
        
        for H in self.homographies:
            # Extract rotation and translation from homography
            h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
            lambda_val = 1.0 / np.linalg.norm(K_inv @ h1)
            
            r1 = lambda_val * K_inv @ h1
            r2 = lambda_val * K_inv @ h2
            r3 = np.cross(r1, r2)  # Third column of rotation matrix
            t = lambda_val * K_inv @ h3
            
            # Combine into [R|t] matrix
            R = np.column_stack([r1, r2, r3])
            extrinsics.append(np.column_stack([R, t]))
        
        return extrinsics
    
    @staticmethod
    def project_with_distortion(params, world_pts, extrinsic):
        """Project 3D points to 2D with radial distortion."""
        fx, fy, skew, cx, cy, k1, k2 = params
        K = np.array([[fx, skew, cx], [0, fy, cy], [0, 0, 1]])
        
        # Transform to camera coordinates
        world_homo = np.column_stack([world_pts, np.ones(len(world_pts))])
        cam_pts = extrinsic @ world_homo.T
        x, y = cam_pts[0] / cam_pts[2], cam_pts[1] / cam_pts[2]  # Normalize
        
        # Apply radial distortion
        r_squared = x**2 + y**2
        distortion_factor = 1 + k1 * r_squared + k2 * r_squared**2
        x_dist = x * distortion_factor
        y_dist = y * distortion_factor
        
        # Convert to pixel coordinates
        pixel_pts = K @ np.vstack([x_dist, y_dist, np.ones(len(x_dist))])
        return pixel_pts[:2].T
    
    def _compute_residuals(self, params, extrinsics):
        """Compute reprojection residuals for optimization."""
        residuals = []
        for world_pts, img_pts, extrinsic in zip(self.object_points, self.image_points, extrinsics):
            projected = self.project_with_distortion(params, world_pts, extrinsic)
            residuals.extend((img_pts - projected).flatten())
        return np.array(residuals)
    
    def refine_parameters(self, K_initial, extrinsics):
        """Refine camera parameters using non-linear optimization."""
        # Initial parameter vector
        initial_params = [
            K_initial[0, 0], K_initial[1, 1], K_initial[0, 1],  # fx, fy, skew
            K_initial[0, 2], K_initial[1, 2],                   # cx, cy
            0.0, 0.0                                            # k1, k2
        ]
        
        # Optimize using Levenberg-Marquardt
        result = least_squares(
            lambda p: self._compute_residuals(p, extrinsics), 
            initial_params, 
            method='lm'
        )
        
        # Extract optimized parameters
        fx, fy, skew, cx, cy, k1, k2 = result.x
        K_final = np.array([[fx, skew, cx], [0, fy, cy], [0, 0, 1]])
        distortion_coeffs = np.array([k1, k2, 0, 0, 0])
        
        return K_final, distortion_coeffs
    
    def _calculate_reprojection_error(self, K, distortion_coeffs, extrinsics):
        """Calculate mean reprojection error across all images."""
        total_error = 0
        total_points = 0
        
        for i, (world_pts, img_pts) in enumerate(zip(self.object_points, self.image_points)):
            R_vec = cv2.Rodrigues(extrinsics[i][:, :3])[0]
            t_vec = extrinsics[i][:, 3]
            projected_pts, _ = cv2.projectPoints(world_pts, R_vec, t_vec, K, distortion_coeffs)
            
            error = cv2.norm(img_pts, projected_pts.reshape(-1, 2), cv2.NORM_L2) / len(projected_pts)
            total_error += error * len(projected_pts)
            total_points += len(projected_pts)
        
        return total_error / total_points
    
    def _process_images(self):
        """Process all images to extract corners and compute homographies."""
        image_files = sorted(glob.glob(os.path.join(self.images_folder, "*.png")))
        valid_count = 0
        for image_path in image_files:
            world_pts, img_pts = self.detect_corners(image_path)
            
            if world_pts is not None:
                self.object_points.append(world_pts)
                self.image_points.append(img_pts)
                
                # Compute homography for this view
                H, _ = cv2.findHomography(world_pts[:, :2], img_pts, cv2.RANSAC)
                self.homographies.append(H)
                valid_count += 1
                print(f"Processed: {os.path.basename(image_path)}")
            else:
                print(f"Failed: {os.path.basename(image_path)}")
        
        if valid_count < 3:
            raise ValueError(f"Only {valid_count} valid images found. Need at least 3 for calibration.")
        
        return valid_count
    
    def save_undistorted_images(self, K, distortion_coeffs):
        result_dir = "results"
        os.makedirs(result_dir, exist_ok=True)
        image_files = sorted(glob.glob(os.path.join(self.images_folder, "*.png")))
        
        for idx, image_path in enumerate(image_files):
            image = cv2.imread(image_path)
            if image is None:
                continue

            undistorted = cv2.undistort(image, K, distortion_coeffs) # Apply undistortion
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY) # Draw corners if detected
            found, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
            if found:
                cv2.drawChessboardCorners(undistorted, self.pattern_size, corners, found)
            
            output_path = os.path.join(result_dir, f"undistorted_{idx+1}.png")
            cv2.imwrite(output_path, undistorted)
    
    def _print_results(self, K, distortion_coeffs, mean_error, valid_images):
        print(f"Images used: {valid_images}")
        print(f"Mean reprojection error: {mean_error:.3f} pixels")
        print(f"\nIntrinsic Matrix (K):")
        print(f"  [{K[0,0]:8.2f} {K[0,1]:8.4f} {K[0,2]:8.2f}]")
        print(f"  [{K[1,0]:8.2f} {K[1,1]:8.2f} {K[1,2]:8.2f}]")
        print(f"  [{K[2,0]:8.2f} {K[2,1]:8.2f} {K[2,2]:8.2f}]")
        print(f"\nCamera Parameters:")
        print(f"  Focal length (fx, fy): ({K[0,0]:.2f}, {K[1,1]:.2f}) pixels")
        print(f"  Principal point (cx, cy): ({K[0,2]:.2f}, {K[1,2]:.2f}) pixels")
        print(f"  Skew coefficient: {K[0,1]:.6f}")
        print(f"\nDistortion Coefficients [k1, k2, p1, p2, k3]:")
        print(f"  {distortion_coeffs}")
    
    def calibrate(self):
        # Step 1: Process images and extract features
        valid_images = self._process_images()
        
        # Step 2: Estimate initial intrinsic parameters
        K_initial = self.extract_intrinsic_matrix()
        
        # Step 3: Calculate extrinsic parameters
        extrinsics = self.calculate_extrinsics(K_initial)
        
        # Step 4: Refine all parameters with non-linear optimization
        K_final, distortion_coeffs = self.refine_parameters(K_initial, extrinsics)
        
        # Step 5: Calculate final reprojection error
        mean_error = self._calculate_reprojection_error(K_final, distortion_coeffs, extrinsics)
        
        # Step 6: Save results and generate output
        self.save_undistorted_images(K_final, distortion_coeffs)
        self._print_results(K_final, distortion_coeffs, mean_error, valid_images)
        
        return K_final, distortion_coeffs, mean_error

def main():
    PATTERN_SIZE = (11, 7)  # Internal corners (width, height)
    SQUARE_SIZE = 30.0      # Square size in mm
    IMAGES_FOLDER = "data"  # Calibration images folder
    
    calibrator = CameraCalibrator(PATTERN_SIZE, SQUARE_SIZE, IMAGES_FOLDER)
    
    try:
        camera_matrix, distortion_coeffs, reprojection_error = calibrator.calibrate()
        print(f"\n Calibration completed successfully!")
        
    except Exception as e:
        print(f"Calibration failed: {str(e)}")

if __name__ == "__main__":
    main()