import cv2
import argparse
import os
import math
import numpy as np

class PanoramaStitching:
    def __init__(self):
        self.matching_threshold = 0.75
        self.ransac_distance_threshold = 7.0  # Pixel distance threshold
        self.ransac_confidence = 0.99
        self.scale = 0.5
        self.final_blend = False
        
    def read_input(self):
        parser = argparse.ArgumentParser(description='Panorama Stitching')
        parser.add_argument('--InputPath', type=str, required=True, help='Path to the input images')
        args = parser.parse_args()
        self.input_path = args.InputPath
        
    def read_images(self):
        files = sorted([f for f in os.listdir(self.input_path) if f.lower().endswith('.jpg')])
        images = []
        for file in files:
            img = cv2.imread(os.path.join(self.input_path, file))
            if img is not None:
                img = cv2.resize(img, (int(img.shape[1] * self.scale), int(img.shape[0] * self.scale)))
                images.append(img)
        return images
            
    def extract_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        return orb.detectAndCompute(gray, None)
    
    def match_features(self, desc1, desc2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < self.matching_threshold * n.distance:
                good_matches.append(m)
        return good_matches
    
    def estimate_homography(self, matches, kp1, kp2):
        if len(matches) < 4:
            return None
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_distance_threshold, confidence=self.ransac_confidence)
        return H
    
    def save_matches_image(self, img1, kp1, img2, kp2, matches, pair_index, iteration=None):
        temp_dir = f'{self.input_path}/temp_pano'
        os.makedirs(temp_dir, exist_ok=True)
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
        
        if iteration is not None:
            filename = f'matches_iter{iteration}_pair{pair_index}.jpg'
        else:
            filename = f'matches_pair{pair_index}.jpg'
            
        save_path = os.path.join(temp_dir, filename)
        cv2.imwrite(save_path, img_matches)
    
    def get_homographies(self, iteration=None):
        homographies = []
        for i in range(len(self.images) - 1):
            kp1, desc1 = self.extract_features(self.images[i])
            kp2, desc2 = self.extract_features(self.images[i + 1])
            matches = self.match_features(desc1, desc2)
            
            self.save_matches_image(self.images[i], kp1, self.images[i + 1], kp2, matches, i, iteration)
            
            H = self.estimate_homography(matches, kp1, kp2)
            if H is not None:
                homographies.append(H)
        return homographies
    
    def create_weighted_mask(self, mask):
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.uint8)
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        max_dist = np.max(dist_transform)
        return dist_transform / max_dist if max_dist > 0 else dist_transform
	
    def stitch_pair(self, homographies):
        for i, H in enumerate(homographies):
            if i >= len(self.images) - 1:
                break
                
            # Determine image order based on position
            n = math.ceil(len(self.images) / 2)
            if i + 1 < n:
                img1, img2 = self.images[i + 1], self.images[i]
            else:
                H = np.linalg.inv(H)
                img1, img2 = self.images[i], self.images[i + 1]
            
            # Calculate canvas size
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            corners2_transformed = cv2.perspectiveTransform(corners2, H)
            all_corners = np.concatenate((corners1, corners2_transformed))
            
            [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
            
            # Translation matrix
            translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
            
            # Warp images
            canvas_size = (x_max - x_min, y_max - y_min)
            img1_warped = cv2.warpPerspective(img1, translation, canvas_size)
            img2_warped = cv2.warpPerspective(img2, translation @ H, canvas_size)
            
            # Create weighted masks and blend
            mask1 = ((img1_warped > 0).astype(np.uint8) * 255)
            mask2 = ((img2_warped > 0).astype(np.uint8) * 255)
            weight1 = self.create_weighted_mask(mask1)[..., None]
            weight2 = self.create_weighted_mask(mask2)[..., None]
            
            denominator = np.where(weight1 + weight2 == 0, 1, weight1 + weight2)
            result = np.clip((img1_warped * weight1 + img2_warped * weight2) / denominator, 0, 255).astype(np.uint8)
            
            # Save result
            if self.final_blend:
                cv2.imwrite(f'{self.input_path}/Final_Panorama.jpg', result)
            else:
                temp_dir = f'{self.input_path}/temp_pano'
                os.makedirs(temp_dir, exist_ok=True)
                cv2.imwrite(f'{temp_dir}/{i}.jpg', result)

    def process_multiple_images(self):
        temp_dir = f'{self.input_path}/temp_pano'
        os.makedirs(temp_dir, exist_ok=True)
        current_images = self.images.copy()
        iteration = 0
        
        while len(current_images) > 2 and iteration < 10:
            print(f"Iteration {iteration}: Processing {len(current_images)} images")
            self.images = current_images
            self.final_blend = False
            
            homographies = self.get_homographies(iteration)
            if not homographies:
                print("No valid homographies found, stopping")
                break
                
            self.stitch_pair(homographies)
            
            # Load intermediate results
            new_images = []
            for i in range(len(homographies)):
                result_path = f'{temp_dir}/{i}.jpg'
                if os.path.exists(result_path):
                    img = cv2.imread(result_path)
                    if img is not None:
                        new_images.append(img)
            
            current_images = new_images
            iteration += 1
        
        # Final stitch of remaining 2 images
        if len(current_images) == 2:
            print("Final stitching of 2 remaining images")
            self.images = current_images
            self.final_blend = True
            homographies = self.get_homographies(iteration)
            if homographies:
                self.stitch_pair(homographies)

    def main(self):
        self.read_input()
        self.images = self.read_images()
            
        if len(self.images) == 2:
            self.final_blend = True
            homographies = self.get_homographies()
            self.stitch_pair(homographies)
        else:
            homographies = self.get_homographies()
            self.stitch_pair(homographies)
            self.process_multiple_images()
        
if __name__ == "__main__":
    PanoramaStitching().main()