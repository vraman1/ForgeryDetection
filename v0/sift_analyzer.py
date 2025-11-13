# sift_analyzer.py - Fully Rewritten
import cv2
import numpy as np
import os
from scipy import ndimage
from scipy.stats import kurtosis, skew

class SIFTAnalyzer:
    def __init__(self):
        self.sift = cv2.SIFT_create(
            nfeatures=5000,
            nOctaveLayers=4,
            contrastThreshold=0.02,
            edgeThreshold=15,
            sigma=1.2
        )
    
    def _enhance_texture(self, gray):
        """Enhanced texture enhancement using multi-scale analysis"""
        # Multi-scale contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12, 12))
        enhanced = clahe.apply(gray)
        
        # Multi-scale texture enhancement
        scales = [3, 5, 7]
        texture_combined = np.zeros_like(enhanced, dtype=np.float32)
        
        for scale in scales:
            # Local variance for texture
            kernel = np.ones((scale, scale), np.float32) / (scale * scale)
            mean = cv2.filter2D(enhanced.astype(np.float32), -1, kernel)
            variance = cv2.filter2D((enhanced.astype(np.float32) ** 2), -1, kernel) - mean ** 2
            texture_combined += variance
            
        # Normalize and combine
        texture_combined = cv2.normalize(texture_combined, None, 0, 255, cv2.NORM_MINMAX)
        final_enhanced = cv2.addWeighted(enhanced, 0.6, texture_combined.astype(np.uint8), 0.4, 0)
        
        return final_enhanced
    
    def _analyze_keypoint_distribution(self, keypoints, image_shape):
        """Comprehensive keypoint distribution analysis"""
        if not keypoints:
            return "No keypoints", {}, []
        
        h, w = image_shape[:2]
        regions = {
            'top_left': 0, 'top_right': 0, 'bottom_left': 0, 'bottom_right': 0,
            'center': 0
        }
        
        center_x, center_y = w // 2, h // 2
        quadrant_threshold = 0.25
        
        for kp in keypoints:
            x, y = kp.pt
            
            # Determine quadrant
            if x < center_x and y < center_y:
                regions['top_left'] += 1
            elif x >= center_x and y < center_y:
                regions['top_right'] += 1
            elif x < center_x and y >= center_y:
                regions['bottom_left'] += 1
            else:
                regions['bottom_right'] += 1
                
            # Check if near center
            if abs(x - center_x) < w * 0.25 and abs(y - center_y) < h * 0.25:
                regions['center'] += 1
        
        total_kp = len(keypoints)
        region_percentages = {region: (count / total_kp) * 100 for region, count in regions.items()}
        
        # Analyze distribution patterns
        max_region = max(regions.values())
        min_region = min(regions.values())
        uniformity_score = (min_region / max_region) if max_region > 0 else 0
        
        if uniformity_score < 0.3:
            distribution_status = "Highly uneven distribution - potential tampering"
        elif uniformity_score < 0.6:
            distribution_status = "Moderately uneven distribution"
        else:
            distribution_status = "Uniform distribution - likely authentic"
        
        return distribution_status, region_percentages, list(regions.values())
    
    def _detect_anomalous_patterns(self, keypoints, descriptors):
        """Detect anomalous patterns in keypoint distribution"""
        if len(keypoints) < 10:
            return ["Insufficient keypoints for pattern analysis"]
        
        anomalies = []
        
        # Analyze spatial clustering
        positions = np.array([kp.pt for kp in keypoints])
        
        # Check for unusual clusters
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=50, min_samples=5).fit(positions)
        unique_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        
        if unique_clusters > 5:
            anomalies.append(f"Multiple dense clusters detected ({unique_clusters} clusters)")
        
        # Check scale distribution
        scales = np.array([kp.size for kp in keypoints])
        scale_std = np.std(scales)
        if scale_std > 30:
            anomalies.append("Highly varied keypoint scales - possible multi-scale tampering")
        
        # Check response strength distribution
        responses = np.array([kp.response for kp in keypoints])
        if np.std(responses) > 0.3:
            anomalies.append("Inconsistent feature strengths detected")
        
        return anomalies
    
    def analyze_image(self, image_path, output_dir="final_results"):
        """Comprehensive SIFT analysis with enhanced visualization"""
        print("🔍 Starting Enhanced SIFT Analysis...")
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Could not load image")
            return 0, None, [], None, {}
        
        original = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing
        enhanced_gray = self._enhance_texture(gray)
        
        # Detect keypoints and descriptors
        keypoints, descriptors = self.sift.detectAndCompute(enhanced_gray, None)
        
        print(f"   ✅ Detected {len(keypoints)} keypoints")
        
        # Comprehensive analysis
        distribution_status, region_stats, region_counts = self._analyze_keypoint_distribution(keypoints, img.shape)
        anomalies = self._detect_anomalous_patterns(keypoints, descriptors)
        
        # Create enhanced visualization
        visualization = self._create_enhanced_visualization(original, keypoints, distribution_status, 
                                                          region_stats, anomalies, len(keypoints))
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"sift_analysis_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, visualization)
        
        print(f"   ✅ SIFT analysis saved: {output_path}")
        print(f"   📊 Distribution: {distribution_status}")
        if anomalies:
            print(f"   ⚠️  Anomalies: {len(anomalies)} detected")
        
        analysis_results = {
            'keypoint_count': len(keypoints),
            'distribution_status': distribution_status,
            'region_statistics': region_stats,
            'anomalies_detected': anomalies,
            'uniformity_score': min(region_counts) / max(region_counts) if max(region_counts) > 0 else 0
        }
        
        return len(keypoints), output_path, keypoints, descriptors, analysis_results
    
    def _create_enhanced_visualization(self, original, keypoints, distribution_status, 
                                     region_stats, anomalies, total_keypoints):
        """Create comprehensive SIFT visualization"""
        # Create base visualization with rich keypoints
        vis = original.copy()
        vis = cv2.drawKeypoints(
            vis, keypoints, None, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            color=(0, 255, 0)  # Green for normal keypoints
        )
        
        h, w = vis.shape[:2]
        
        # Add quadrant lines
        #cv2.line(vis, (w//2, 0), (w//2, h), (255, 255, 255), 2)
        #cv2.line(vis, (0, h//2), (w, h//2), (255, 255, 255), 2)
        
        # Add information overlay
        y_offset = 40
        #cv2.putText(vis, "SIFT FEATURE ANALYSIS", (20, y_offset), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 35
        
        #cv2.putText(vis, f"Total Keypoints: {total_keypoints}", (20, y_offset), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        # Add distribution status
        status_color = (0, 0, 255) if "tampering" in distribution_status.lower() else (0, 255, 0)
        #cv2.putText(vis, f"Distribution: {distribution_status}", (20, y_offset), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        y_offset += 25
        
        # Add quadrant statistics
        #cv2.putText(vis, "Quadrant Distribution:", (20, y_offset), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        for region, percentage in list(region_stats.items())[:4]:  # Only show quadrants
            #cv2.putText(vis, f"{region}: {percentage:.1f}%", (30, y_offset), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 18
        
        # Add anomalies if any
        if anomalies:
            y_offset += 10
            #cv2.putText(vis, f"Anomalies Detected: {len(anomalies)}", (20, y_offset), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            y_offset += 20
            
            for i, anomaly in enumerate(anomalies[:2]):  # Show first 2 anomalies
                if y_offset < h - 30:
                   # cv2.putText(vis, f"- {anomaly[:40]}...", (25, y_offset), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                    y_offset += 16
        
        # Add border based on analysis results
        #border_color = (0, 0, 255) if "tampering" in distribution_status.lower() else (0, 255, 0)
        #border_thickness = 4 if "tampering" in distribution_status.lower() else 2
        #cv2.rectangle(vis, (5, 5), (w-5, h-5), border_color, border_thickness)
        
        return vis

# Compatibility functions
def enhanced_sift_analysis(image_path, output_dir="final_results", use_entropy=True):
    analyzer = SIFTAnalyzer()
    return analyzer.analyze_image(image_path, output_dir)

def analyze_keypoint_distribution(keypoints, image_shape):
    analyzer = SIFTAnalyzer()
    distribution_status, _, _ = analyzer._analyze_keypoint_distribution(keypoints, image_shape)
    return distribution_status