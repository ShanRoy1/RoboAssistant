#!/usr/bin/env python3
# RoboAssistant - Fusion QC Module
# Combines LIDAR and RealSense data to create high-precision scans for QC comparison with Fusion 360 models

import os
import time
import logging
import numpy as np
import open3d as o3d
import trimesh
import json
import math
import cv2
from pathlib import Path

logger = logging.getLogger("RoboAssistant.FusionQC")

class FusionQC:
    """QC system that combines LIDAR and depth camera data and compares to CAD models"""
    
    def __init__(self, depth_camera=None, lidar_scanner=None):
        """Initialize the Fusion QC module
        
        Args:
            depth_camera: DepthCamera instance
            lidar_scanner: LidarScanner instance
        """
        logger.info("Initializing Fusion QC module")
        self.depth_camera = depth_camera
        self.lidar_scanner = lidar_scanner
        
        # Calibration matrices for sensor fusion
        self.lidar_to_camera_transform = np.eye(4)  # Identity matrix as default
        self.camera_intrinsics = None
        
        # Data storage
        self.combined_pointcloud = None
        self.reference_model = None
        
        # QC tolerances
        self.distance_tolerance = 0.002  # 2mm default tolerance
        self.outlier_threshold = 0.01    # 1cm threshold for outliers
        
        # Results
        self.last_comparison_result = None
        
        # Create output directories
        os.makedirs("output/scans", exist_ok=True)
        os.makedirs("output/reports", exist_ok=True)
        os.makedirs("output/visualizations", exist_ok=True)
    
    def load_calibration(self, calibration_file):
        """Load sensor calibration data from file
        
        Args:
            calibration_file (str): Path to calibration JSON file
            
        Returns:
            bool: Success status
        """
        try:
            with open(calibration_file, 'r') as f:
                calibration_data = json.load(f)
            
            # Load transformation matrix
            if 'lidar_to_camera_transform' in calibration_data:
                self.lidar_to_camera_transform = np.array(calibration_data['lidar_to_camera_transform'])
            
            # Load camera intrinsics
            if 'camera_intrinsics' in calibration_data:
                self.camera_intrinsics = np.array(calibration_data['camera_intrinsics'])
            
            logger.info(f"Loaded calibration from {calibration_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration: {str(e)}")
            return False
    
    def save_calibration(self, calibration_file):
        """Save current calibration data to file
        
        Args:
            calibration_file (str): Path to save calibration JSON file
            
        Returns:
            bool: Success status
        """
        try:
            calibration_data = {
                'lidar_to_camera_transform': self.lidar_to_camera_transform.tolist(),
                'camera_intrinsics': self.camera_intrinsics.tolist() if self.camera_intrinsics is not None else None
            }
            
            with open(calibration_file, 'w') as f:
                json.dump(calibration_data, f, indent=4)
            
            logger.info(f"Saved calibration to {calibration_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save calibration: {str(e)}")
            return False
    
    def calibrate_sensors(self):
        """Run automatic calibration routine to align LIDAR and camera data
        
        Returns:
            bool: Success status
        """
        if not self.depth_camera or not self.lidar_scanner:
            logger.error("Both depth camera and LIDAR are required for calibration")
            return False
        
        logger.info("Starting sensor calibration")
        try:
            # Start both sensors
            self.depth_camera.start_streaming()
            self.lidar_scanner.start_scanning()
            
            # Wait for sensors to stabilize
            time.sleep(2)
            
            # Get data from both sensors
            color, depth = self.depth_camera.get_frames()
            lidar_scan = self.lidar_scanner.get_latest_scan()
            
            if depth is None or lidar_scan is None:
                logger.error("Failed to get data from sensors")
                return False
            
            # Convert depth to point cloud
            camera_points = self._depth_to_pointcloud(depth)
            
            # Convert LIDAR scan to point cloud
            lidar_points = self._lidar_to_pointcloud(lidar_scan)
            
            # Use point-to-plane ICP to align the point clouds
            # This is a simplified placeholder - actual implementation would require more sophisticated alignment
            source = o3d.geometry.PointCloud()
            source.points = o3d.utility.Vector3dVector(lidar_points)
            
            target = o3d.geometry.PointCloud()
            target.points = o3d.utility.Vector3dVector(camera_points)
            
            # Estimate normals for point-to-plane ICP
            target.estimate_normals()
            
            # Run ICP
            icp_result = o3d.pipelines.registration.registration_icp(
                source, target, 
                max_correspondence_distance=0.05,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            
            # Update calibration transform
            self.lidar_to_camera_transform = icp_result.transformation
            
            # Save calibration to file
            self.save_calibration("config/sensor_calibration.json")
            
            logger.info("Calibration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {str(e)}")
            return False
        finally:
            # Stop sensors
            if self.depth_camera:
                self.depth_camera.stop_streaming()
            if self.lidar_scanner:
                self.lidar_scanner.stop_scanning()
    
    def create_combined_scan(self, scan_duration=15, high_quality=False):
        """Create a combined point cloud from both sensors
        
        Args:
            scan_duration (int): Duration of scan in seconds
            high_quality (bool): If True, perform multiple scans and merge
            
        Returns:
            bool: Success status
        """
        if not self.depth_camera or not self.lidar_scanner:
            logger.error("Both depth camera and LIDAR are required for combined scan")
            return False
        
        logger.info(f"Starting combined scan (duration: {scan_duration}s, high quality: {high_quality})")
        
        try:
            # Start both sensors
            self.depth_camera.start_streaming()
            self.lidar_scanner.start_scanning()
            
            # Wait for sensors to stabilize
            time.sleep(2)
            
            camera_pointclouds = []
            lidar_pointclouds = []
            
            # Number of scans to take
            num_scans = 5 if high_quality else 1
            
            # Collect multiple scans if high quality is requested
            scan_interval = scan_duration / num_scans
            
            for i in range(num_scans):
                # Get depth camera frame
                _, depth = self.depth_camera.get_frames()
                if depth is not None:
                    camera_points = self._depth_to_pointcloud(depth)
                    camera_pointclouds.append(camera_points)
                
                # Get LIDAR scan
                lidar_scan = self.lidar_scanner.get_latest_scan()
                if lidar_scan is not None:
                    lidar_points = self._lidar_to_pointcloud(lidar_scan)
                    lidar_pointclouds.append(lidar_points)
                
                # Wait for next scan if not the last one
                if i < num_scans - 1:
                    time.sleep(scan_interval)
            
            # Combine all scans
            combined_camera_points = np.vstack(camera_pointclouds) if camera_pointclouds else None
            combined_lidar_points = np.vstack(lidar_pointclouds) if lidar_pointclouds else None
            
            if combined_camera_points is None or combined_lidar_points is None:
                logger.error("Failed to collect point cloud data")
                return False
            
            # Transform LIDAR points to camera coordinate system
            transformed_lidar_points = self._transform_points(combined_lidar_points, self.lidar_to_camera_transform)
            
            # Merge the point clouds
            all_points = np.vstack([combined_camera_points, transformed_lidar_points])
            
            # Create Open3D point cloud
            self.combined_pointcloud = o3d.geometry.PointCloud()
            self.combined_pointcloud.points = o3d.utility.Vector3dVector(all_points)
            
            # Apply outlier removal for cleaner point cloud
            self.combined_pointcloud = self.combined_pointcloud.voxel_down_sample(voxel_size=0.005)  # 5mm voxel size
            self.combined_pointcloud, _ = self.combined_pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # Save the point cloud
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"output/scans/combined_scan_{timestamp}.ply"
            o3d.io.write_point_cloud(output_file, self.combined_pointcloud)
            
            logger.info(f"Combined scan completed and saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Combined scan failed: {str(e)}")
            return False
        finally:
            # Stop sensors
            if self.depth_camera:
                self.depth_camera.stop_streaming()
            if self.lidar_scanner:
                self.lidar_scanner.stop_scanning()
    
    def load_fusion360_model(self, model_file):
        """Load a Fusion 360 model for comparison
        
        Args:
            model_file (str): Path to model file (STL, OBJ, etc.)
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Loading Fusion 360 model from {model_file}")
            
            # Load the model using trimesh
            self.reference_model = trimesh.load(model_file)
            
            # Convert to Open3D mesh for easier processing
            vertices = np.array(self.reference_model.vertices)
            faces = np.array(self.reference_model.faces)
            
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            
            self.reference_mesh = mesh
            
            logger.info(f"Successfully loaded model with {len(vertices)} vertices and {len(faces)} faces")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def align_scan_to_model(self, initial_transform=None):
        """Align the scan point cloud to the reference model
        
        Args:
            initial_transform (numpy.ndarray): Initial 4x4 transformation matrix guess
            
        Returns:
            numpy.ndarray: The final transformation matrix
        """
        if self.combined_pointcloud is None or self.reference_mesh is None:
            logger.error("Both scan and reference model must be loaded before alignment")
            return None
        
        try:
            logger.info("Aligning scan to reference model")
            
            # Sample points from the reference mesh
            reference_points = self.reference_mesh.sample_points_uniformly(number_of_points=100000)
            
            # Use point-to-point ICP for initial alignment
            if initial_transform is None:
                initial_transform = np.eye(4)
            
            # Apply initial transform to the scan
            scan_points = copy.deepcopy(self.combined_pointcloud)
            scan_points.transform(initial_transform)
            
            # Run global registration first for rough alignment
            # This uses FPFH features for better initial alignment
            scan_points.estimate_normals()
            reference_points.estimate_normals()
            
            scan_fpfh = o3d.pipelines.registration.compute_fpfh_feature(scan_points, 
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
            
            reference_fpfh = o3d.pipelines.registration.compute_fpfh_feature(reference_points,
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
            
            # Use RANSAC for global registration
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                scan_points, reference_points, scan_fpfh, reference_fpfh, 
                mutual_filter=True,
                max_correspondence_distance=0.05,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                ransac_n=3,
                checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
            )
            
            # Refine with ICP
            result_icp = o3d.pipelines.registration.registration_icp(
                scan_points, reference_points, 0.02, result_ransac.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            
            # Apply the transformation to the combined pointcloud
            self.combined_pointcloud.transform(result_icp.transformation)
            
            logger.info(f"Alignment complete with fitness score: {result_icp.fitness}")
            return result_icp.transformation
            
        except Exception as e:
            logger.error(f"Alignment failed: {str(e)}")
            return None
    
    def compare_with_model(self):
        """Compare the aligned scan with the reference model
        
        Returns:
            dict: Comparison results and metrics
        """
        if self.combined_pointcloud is None or self.reference_mesh is None:
            logger.error("Both scan and reference model must be loaded and aligned before comparison")
            return None
        
        try:
            logger.info("Comparing scan to reference model")
            
            # Sample points from the reference mesh
            reference_points = self.reference_mesh.sample_points_uniformly(number_of_points=100000)
            
            # Calculate distances from scan to reference model
            distances = np.asarray(self.combined_pointcloud.compute_point_cloud_distance(reference_points))
            
            # Analyze the distances
            mean_distance = np.mean(distances)
            max_distance = np.max(distances)
            std_distance = np.std(distances)
            
            # Count points outside of tolerance
            points_outside_tolerance = np.sum(distances > self.distance_tolerance)
            percentage_outside = (points_outside_tolerance / len(distances)) * 100
            
            # Generate results
            result = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'mean_distance': float(mean_distance),
                'max_distance': float(max_distance),
                'std_distance': float(std_distance),
                'points_outside_tolerance': int(points_outside_tolerance),
                'percentage_outside': float(percentage_outside),
                'total_points': int(len(distances)),
                'tolerance': float(self.distance_tolerance),
                'status': 'PASS' if percentage_outside < 5.0 else 'FAIL'
            }
            
            # Store the result
            self.last_comparison_result = result
            
            # Save results to file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_file = f"output/reports/qc_report_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(result, f, indent=4)
            
            # Create visualization with color-coded distances
            self._create_comparison_visualization(distances, timestamp)
            
            logger.info(f"Comparison complete. Status: {result['status']}")
            logger.info(f"Mean distance: {mean_distance:.4f}m, Max: {max_distance:.4f}m")
            logger.info(f"Points outside tolerance: {points_outside_tolerance} ({percentage_outside:.2f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Comparison failed: {str(e)}")
            return None
    
    def _create_comparison_visualization(self, distances, timestamp):
        """Create color-coded visualization of comparison results
        
        Args:
            distances (numpy.ndarray): Point distances from comparison
            timestamp (str): Timestamp for output filename
        """
        try:
            # Normalize distances for coloring
            colors = np.zeros((len(distances), 3))
            
            # Color scheme: Blue (0,0,1) for perfect match, Red (1,0,0) for max deviation
            for i, d in enumerate(distances):
                if d <= self.distance_tolerance:
                    # Good match - blue to green gradient
                    ratio = d / self.distance_tolerance
                    colors[i] = [0, ratio, 1-ratio]
                else:
                    # Outside tolerance - green to red gradient
                    if d >= self.outlier_threshold:
                        colors[i] = [1, 0, 0]  # Red for outliers
                    else:
                        # Gradient from green to red
                        ratio = (d - self.distance_tolerance) / (self.outlier_threshold - self.distance_tolerance)
                        colors[i] = [ratio, 1-ratio, 0]
            
            # Create colored point cloud
            colored_cloud = o3d.geometry.PointCloud()
            colored_cloud.points = self.combined_pointcloud.points
            colored_cloud.colors = o3d.utility.Vector3dVector(colors)
            
            # Save visualization
            vis_file = f"output/visualizations/qc_vis_{timestamp}.ply"
            o3d.io.write_point_cloud(vis_file, colored_cloud)
            
            logger.info(f"Created visualization at {vis_file}")
        except Exception as e:
            logger.error(f"Failed to create visualization: {str(e)}")
    
    def _depth_to_pointcloud(self, depth_image):
        """Convert depth image to 3D point cloud
        
        Args:
            depth_image (numpy.ndarray): Depth image from RealSense
            
        Returns:
            numpy.ndarray: Nx3 array of 3D points
        """
        # This is a simplified conversion - in reality would use actual camera intrinsics
        height, width = depth_image.shape
        
        # Use camera intrinsics if available, otherwise use defaults
        if self.camera_intrinsics is not None:
            fx = self.camera_intrinsics[0, 0]
            fy = self.camera_intrinsics[1, 1]
            cx = self.camera_intrinsics[0, 2]
            cy = self.camera_intrinsics[1, 2]
        else:
            # Default intrinsics (approximate values for RealSense)
            fx = 600.0
            fy = 600.0
            cx = width / 2
            cy = height / 2
        
        # Create a grid of image coordinates
        v, u = np.indices(depth_image.shape)
        
        # Convert depth to meters from mm
        z = depth_image * 0.001
        
        # Valid depth points only
        valid = (z > 0)
        
        # Calculate x and y coordinates
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack coordinates and filter valid points
        points = np.stack((x[valid], y[valid], z[valid]), axis=-1)
        
        return points
    
    def _lidar_to_pointcloud(self, lidar_scan):
        """Convert LIDAR scan to 3D point cloud
        
        Args:
            lidar_scan (list): LIDAR scan data (angle, distance, quality)
            
        Returns:
            numpy.ndarray: Nx3 array of 3D points
        """
        points = []
        
        for angle, distance, quality in lidar_scan:
            # Filter out low-quality points
            if quality < 10 or distance <= 0:
                continue
                
            # Convert polar to Cartesian coordinates
            rad = math.radians(angle)
            x = distance * math.cos(rad)
            y = distance * math.sin(rad)
            z = 0.0  # LIDAR scan is on a 2D plane
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def _transform_points(self, points, transform):
        """Apply transformation to points
        
        Args:
            points (numpy.ndarray): Nx3 array of points
            transform (numpy.ndarray): 4x4 transformation matrix
            
        Returns:
            numpy.ndarray: Transformed points
        """
        # Homogeneous coordinates
        homogeneous = np.ones((points.shape[0], 4))
        homogeneous[:, :3] = points
        
        # Apply transformation
        transformed = np.dot(homogeneous, transform.T)
        
        # Back to 3D
        return transformed[:, :3]

# Example usage if run directly
if __name__ == "__main__":
    from depth_camera import DepthCamera
    from lidar_scanner import LidarScanner
    import time
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize hardware
    depth_camera = DepthCamera()
    lidar_scanner = LidarScanner()
    
    if depth_camera.connect() and lidar_scanner.connect():
        # Create QC system
        qc = FusionQC(depth_camera, lidar_scanner)
        
        # Try to load existing calibration
        if not qc.load_calibration("config/sensor_calibration.json"):
            print("No calibration found, performing new calibration...")
            qc.calibrate_sensors()
        
        # Create combined scan
        print("Creating combined scan (15s)...")
        qc.create_combined_scan(scan_duration=15, high_quality=True)
        
        # Load a reference model (would typically be exported from Fusion 360)
        if qc.load_fusion360_model("path_to_reference_model.stl"):
            # Align scan to model
            qc.align_scan_to_model()
            
            # Compare with reference model
            result = qc.compare_with_model()
            
            # Print results
            if result:
                print(f"QC Result: {result['status']}")
                print(f"Mean deviation: {result['mean_distance']*1000:.2f}mm")
                print(f"Points outside tolerance: {result['percentage_outside']:.2f}%")
        
        # Cleanup
        depth_camera.disconnect()
        lidar_scanner.disconnect()
    else:
        print("Failed to connect to sensors") 