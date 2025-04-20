#!/usr/bin/env python3
# RoboAssistant - Depth Camera Module
# Interface for the Intel RealSense D455 camera

import time
import logging
import numpy as np
import threading
import cv2
import os

# This would normally import the Intel RealSense library
# import pyrealsense2 as rs

logger = logging.getLogger("RoboAssistant.DepthCamera")

class DepthCamera:
    """Interface for the Intel RealSense D455 depth camera"""
    
    def __init__(self):
        """Initialize the depth camera"""
        logger.info("Initializing Intel RealSense D455 depth camera")
        
        self.connected = False
        self.streaming = False
        
        # Camera objects (will be initialized during connection)
        self.pipeline = None
        self.config = None
        self.align = None
        
        # Frame data
        self.color_frame = None
        self.depth_frame = None
        self.points = None
        
        # Processing settings
        self.depth_scale = 0.001  # meters per depth unit
        self.depth_min = 0.3      # minimum depth in meters
        self.depth_max = 10.0     # maximum depth in meters
        
        # Thread for background processing
        self.processing_thread = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
    
    def connect(self):
        """Connect to the depth camera and initialize the pipeline"""
        logger.info("Connecting to Intel RealSense D455")
        
        try:
            # In a real implementation, this would use the actual RealSense SDK
            # self.pipeline = rs.pipeline()
            # self.config = rs.config()
            # self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            # self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            # self.align = rs.align(rs.stream.color)
            # self.pipeline.start(self.config)
            
            # For now, we'll simulate a successful connection
            self.connected = True
            logger.info("Successfully connected to depth camera")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to depth camera: {str(e)}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the depth camera"""
        if self.connected:
            logger.info("Disconnecting from depth camera")
            
            # Stop streaming if active
            if self.streaming:
                self.stop_streaming()
            
            # In a real implementation, this would stop the RealSense pipeline
            # if self.pipeline:
            #     self.pipeline.stop()
            
            self.connected = False
            logger.info("Disconnected from depth camera")
            return True
        return False
    
    def start_streaming(self):
        """Start continuous streaming and processing of depth data"""
        if not self.connected:
            logger.error("Cannot start streaming: Not connected to depth camera")
            return False
        
        if self.streaming:
            logger.warning("Streaming is already active")
            return True
        
        logger.info("Starting depth camera streaming")
        self.streaming = True
        self.stop_event.clear()
        
        # Start processing in a background thread
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Depth camera streaming started")
        return True
    
    def stop_streaming(self):
        """Stop the continuous streaming of depth data"""
        if not self.streaming:
            return False
        
        logger.info("Stopping depth camera streaming")
        self.stop_event.set()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=3.0)
            if self.processing_thread.is_alive():
                logger.warning("Processing thread did not terminate properly")
        
        self.streaming = False
        logger.info("Depth camera streaming stopped")
        return True
    
    def _process_frames(self):
        """Background thread for continuous frame processing"""
        logger.info("Frame processing thread started")
        
        try:
            while not self.stop_event.is_set():
                # Simulate capturing and processing frames
                # In a real implementation, this would get frames from the RealSense SDK
                # frames = self.pipeline.wait_for_frames()
                # aligned_frames = self.align.process(frames)
                # color_frame = aligned_frames.get_color_frame()
                # depth_frame = aligned_frames.get_depth_frame()
                
                # Simulate random depth and color data for testing
                color_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                depth_data = np.random.randint(0, 10000, (480, 640), dtype=np.uint16)
                
                with self.lock:
                    self.color_frame = color_data
                    self.depth_frame = depth_data
                
                # Process at approximately 30 FPS
                time.sleep(1/30)
        
        except Exception as e:
            logger.error(f"Error in frame processing thread: {str(e)}")
        finally:
            logger.info("Frame processing thread stopped")
    
    def get_frames(self):
        """Get the latest color and depth frames
        
        Returns:
            tuple: (color_frame, depth_frame) as numpy arrays, or (None, None) if not available
        """
        if not self.streaming:
            logger.warning("Cannot get frames: Streaming is not active")
            return None, None
        
        with self.lock:
            color = self.color_frame.copy() if self.color_frame is not None else None
            depth = self.depth_frame.copy() if self.depth_frame is not None else None
        
        return color, depth
    
    def get_depth_at_point(self, x, y):
        """Get the depth value at a specific point in the image
        
        Args:
            x (int): X coordinate in the image
            y (int): Y coordinate in the image
        
        Returns:
            float: Depth in meters, or None if not available
        """
        if not self.streaming or self.depth_frame is None:
            logger.warning("Cannot get depth: Streaming is not active or no frame available")
            return None
        
        with self.lock:
            if 0 <= x < self.depth_frame.shape[1] and 0 <= y < self.depth_frame.shape[0]:
                # Convert depth value to meters
                depth_value = self.depth_frame[y, x] * self.depth_scale
                if self.depth_min <= depth_value <= self.depth_max:
                    return depth_value
                else:
                    return None
            else:
                logger.warning(f"Coordinates ({x}, {y}) out of bounds")
                return None
    
    def save_point_cloud(self, filename):
        """Save the current 3D point cloud to a file
        
        Args:
            filename (str): Filename to save the point cloud (PLY format)
        
        Returns:
            bool: Success status
        """
        if not self.streaming:
            logger.warning("Cannot save point cloud: Streaming is not active")
            return False
        
        try:
            color, depth = self.get_frames()
            if color is None or depth is None:
                logger.warning("No frames available to save point cloud")
                return False
            
            # Simulate point cloud generation
            logger.info(f"Generating point cloud and saving to {filename}")
            
            # In a real implementation, this would use the RealSense SDK to generate a point cloud
            # pc = rs.pointcloud()
            # pc.map_to(color_frame)
            # points = pc.calculate(depth_frame)
            # points.export_to_ply(filename, color_frame)
            
            # For now, just create a dummy PLY file
            with open(filename, 'w') as f:
                f.write("""ply
format ascii 1.0
element vertex 10
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
0.0 0.0 0.0 255 0 0
0.1 0.0 0.0 0 255 0
0.0 0.1 0.0 0 0 255
0.1 0.1 0.0 255 255 0
0.0 0.0 0.1 255 0 255
0.1 0.0 0.1 0 255 255
0.0 0.1 0.1 128 128 128
0.1 0.1 0.1 255 255 255
0.05 0.05 0.0 100 100 100
0.05 0.05 0.1 200 200 200
""")
            
            logger.info(f"Point cloud saved to {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save point cloud: {str(e)}")
            return False
    
    def detect_objects(self):
        """Detect objects in the current color and depth frames
        
        Returns:
            list: List of detected objects with their positions and dimensions
        """
        if not self.streaming:
            logger.warning("Cannot detect objects: Streaming is not active")
            return []
        
        try:
            color, depth = self.get_frames()
            if color is None or depth is None:
                logger.warning("No frames available to detect objects")
                return []
            
            # Simulate object detection
            # In a real implementation, this would use OpenCV or a neural network
            # for object detection and depth information for 3D positioning
            
            # For testing, just return a simulated object
            objects = [
                {
                    "label": "cup",
                    "confidence": 0.92,
                    "position": (0.2, 0.1, 0.5),  # x, y, z in meters
                    "dimensions": (0.1, 0.1, 0.2)  # width, height, depth in meters
                }
            ]
            
            return objects
        
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            return []
    
    def calculate_distance_between_points(self, point1, point2):
        """Calculate the 3D distance between two points in the image
        
        Args:
            point1 (tuple): (x, y) coordinates of the first point
            point2 (tuple): (x, y) coordinates of the second point
        
        Returns:
            float: 3D distance in meters, or None if not available
        """
        x1, y1 = point1
        x2, y2 = point2
        
        depth1 = self.get_depth_at_point(x1, y1)
        depth2 = self.get_depth_at_point(x2, y2)
        
        if depth1 is None or depth2 is None:
            return None
        
        # Calculate 3D coordinates (simplified)
        # In a real implementation, this would use the camera intrinsics
        z1 = depth1
        z2 = depth2
        x1_3d = (x1 - 320) * z1 / 600
        y1_3d = (y1 - 240) * z1 / 600
        x2_3d = (x2 - 320) * z2 / 600
        y2_3d = (y2 - 240) * z2 / 600
        
        # Calculate Euclidean distance
        distance = np.sqrt((x2_3d - x1_3d)**2 + (y2_3d - y1_3d)**2 + (z2 - z1)**2)
        return distance

# Test function
def test_depth_camera():
    """Simple test function for the depth camera"""
    camera = DepthCamera()
    if camera.connect():
        print("Connected to depth camera")
        
        if camera.start_streaming():
            print("Started streaming")
            
            # Wait for a few frames to be captured
            time.sleep(2)
            
            # Get and display a frame
            color, depth = camera.get_frames()
            if color is not None and depth is not None:
                print(f"Color frame shape: {color.shape}")
                print(f"Depth frame shape: {depth.shape}")
                
                # Save frames for testing
                os.makedirs("../logs", exist_ok=True)
                cv2.imwrite("../logs/color_test.jpg", color)
                # Create a normalized depth visualization
                depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite("../logs/depth_test.jpg", depth_vis)
                
                print("Frames saved to logs directory")
            
            # Save a point cloud
            camera.save_point_cloud("../logs/test_pointcloud.ply")
            
            # Stop streaming
            camera.stop_streaming()
        
        camera.disconnect()
        print("Test completed")
    else:
        print("Failed to connect to depth camera")

if __name__ == "__main__":
    # Set up logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_depth_camera() 