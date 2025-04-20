#!/usr/bin/env python3
# RoboAssistant - LIDAR Scanner Module
# Interface for the RPLIDAR A1M8 scanner

import time
import logging
import numpy as np
import threading
import math
import matplotlib.pyplot as plt
import os

# This would normally import the RPLIDAR library
# from rplidar import RPLidar

logger = logging.getLogger("RoboAssistant.LidarScanner")

class LidarScanner:
    """Interface for the RPLIDAR A1M8 scanner"""
    
    def __init__(self, port="/dev/ttyUSB0"):
        """Initialize the LIDAR scanner
        
        Args:
            port (str): Serial port for the LIDAR scanner
        """
        logger.info(f"Initializing RPLIDAR A1M8 scanner on port {port}")
        
        self.port = port
        self.lidar = None
        self.connected = False
        self.scanning = False
        
        # Scan data
        self.scan_data = []
        self.latest_scan = None
        
        # Thread for continuous scanning
        self.scan_thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        
        # Scan settings
        self.max_distance = 12.0  # maximum distance to consider (meters)
        self.angular_resolution = 1.0  # angular resolution in degrees
    
    def connect(self):
        """Connect to the LIDAR scanner"""
        logger.info(f"Connecting to LIDAR scanner on {self.port}")
        
        try:
            # In a real implementation, this would use the actual RPLIDAR library
            # self.lidar = RPLidar(self.port)
            # info = self.lidar.get_info()
            # health = self.lidar.get_health()
            # logger.info(f"LIDAR info: {info}")
            # logger.info(f"LIDAR health: {health}")
            
            # For now, we'll simulate a successful connection
            self.connected = True
            logger.info("Successfully connected to LIDAR scanner")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to LIDAR scanner: {str(e)}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the LIDAR scanner"""
        if self.connected:
            logger.info("Disconnecting from LIDAR scanner")
            
            # Stop scanning if active
            if self.scanning:
                self.stop_scanning()
            
            # In a real implementation, this would stop and disconnect the RPLIDAR
            # if self.lidar:
            #     self.lidar.stop()
            #     self.lidar.stop_motor()
            #     self.lidar.disconnect()
            
            self.connected = False
            logger.info("Disconnected from LIDAR scanner")
            return True
        return False
    
    def start_scanning(self):
        """Start continuous scanning"""
        if not self.connected:
            logger.error("Cannot start scanning: Not connected to LIDAR scanner")
            return False
        
        if self.scanning:
            logger.warning("Scanning is already active")
            return True
        
        logger.info("Starting LIDAR scanning")
        self.scanning = True
        self.stop_event.clear()
        
        # Start scanning in a background thread
        self.scan_thread = threading.Thread(target=self._scan_thread)
        self.scan_thread.daemon = True
        self.scan_thread.start()
        
        logger.info("LIDAR scanning started")
        return True
    
    def stop_scanning(self):
        """Stop the continuous scanning"""
        if not self.scanning:
            return False
        
        logger.info("Stopping LIDAR scanning")
        self.stop_event.set()
        
        if self.scan_thread:
            self.scan_thread.join(timeout=3.0)
            if self.scan_thread.is_alive():
                logger.warning("Scan thread did not terminate properly")
        
        self.scanning = False
        logger.info("LIDAR scanning stopped")
        return True
    
    def _scan_thread(self):
        """Background thread for continuous scanning"""
        logger.info("Scan thread started")
        
        try:
            # In a real implementation, this would use the RPLIDAR SDK
            # self.lidar.start_motor()
            # for scan in self.lidar.iter_scans():
            #     if self.stop_event.is_set():
            #         break
            #     with self.lock:
            #         self.latest_scan = scan
            #         self.process_scan(scan)
            
            # Simulate scanning by generating random data
            while not self.stop_event.is_set():
                # Generate a simulated scan (angle, distance, quality)
                simulated_scan = []
                for angle in range(0, 360, int(self.angular_resolution)):
                    # Random distance between 0.1 and max_distance meters
                    distance = np.random.uniform(0.1, self.max_distance)
                    quality = np.random.randint(0, 100)
                    simulated_scan.append((angle, distance, quality))
                
                with self.lock:
                    self.latest_scan = simulated_scan
                    self.process_scan(simulated_scan)
                
                # Simulate scan rate of about 5 Hz
                time.sleep(0.2)
        
        except Exception as e:
            logger.error(f"Error in scan thread: {str(e)}")
        finally:
            # In a real implementation, this would stop the motor
            # self.lidar.stop_motor()
            logger.info("Scan thread stopped")
    
    def process_scan(self, scan):
        """Process a new scan and update internal data
        
        Args:
            scan (list): List of (angle, distance, quality) tuples
        """
        # Keep the scan data history limited to last 10 scans
        self.scan_data.append(scan)
        if len(self.scan_data) > 10:
            self.scan_data.pop(0)
    
    def get_latest_scan(self):
        """Get the latest scan data
        
        Returns:
            list: List of (angle, distance, quality) tuples, or None if not available
        """
        if not self.scanning:
            logger.warning("Cannot get scan data: Scanning is not active")
            return None
        
        with self.lock:
            return self.latest_scan.copy() if self.latest_scan else None
    
    def get_obstacle_map(self):
        """Generate a 2D obstacle map from the latest scan
        
        Returns:
            numpy.ndarray: 2D boolean array (True=occupied, False=free)
        """
        scan = self.get_latest_scan()
        if scan is None:
            return None
        
        # Create a 2D grid centered on the LIDAR
        grid_size = 100  # 100x100 grid
        resolution = 0.1  # 10 cm per cell
        grid = np.zeros((grid_size, grid_size), dtype=bool)
        
        # Convert polar coordinates (angle, distance) to Cartesian (x, y)
        center = grid_size // 2
        for angle, distance, _ in scan:
            if distance > self.max_distance:
                continue
                
            # Convert to radians and calculate x, y
            rad = math.radians(angle)
            x = distance * math.cos(rad)
            y = distance * math.sin(rad)
            
            # Convert to grid coordinates
            grid_x = int(center + x / resolution)
            grid_y = int(center + y / resolution)
            
            # Check if within grid bounds
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                grid[grid_y, grid_x] = True
        
        return grid
    
    def visualize_scan(self, filename=None):
        """Visualize the latest scan data
        
        Args:
            filename (str, optional): If provided, save the visualization to this file
        
        Returns:
            bool: Success status
        """
        scan = self.get_latest_scan()
        if scan is None:
            logger.warning("No scan data available to visualize")
            return False
        
        try:
            # Create polar plot
            plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, projection='polar')
            
            # Extract angles and distances
            angles = np.radians([point[0] for point in scan])
            distances = [point[1] for point in scan]
            
            # Plot the scan points
            ax.scatter(angles, distances, s=5, c='blue', alpha=0.5)
            
            # Set plot limits and labels
            ax.set_rmax(self.max_distance)
            ax.set_title('LIDAR Scan', fontsize=14)
            ax.grid(True)
            
            # Save or show the plot
            if filename:
                plt.savefig(filename)
                logger.info(f"Scan visualization saved to {filename}")
            else:
                plt.show()
            
            plt.close()
            return True
        
        except Exception as e:
            logger.error(f"Failed to visualize scan: {str(e)}")
            return False
    
    def detect_obstacles(self, min_distance=0.5):
        """Detect obstacles in the scan data
        
        Args:
            min_distance (float): Minimum distance to consider as an obstacle (meters)
        
        Returns:
            list: List of obstacles as (angle, distance) tuples
        """
        scan = self.get_latest_scan()
        if scan is None:
            logger.warning("No scan data available to detect obstacles")
            return []
        
        obstacles = []
        for angle, distance, quality in scan:
            if distance < min_distance and quality > 10:
                obstacles.append((angle, distance))
        
        return obstacles
    
    def get_nearest_obstacle(self):
        """Get the nearest obstacle from the latest scan
        
        Returns:
            tuple: (angle, distance) of the nearest obstacle, or None if no obstacles
        """
        scan = self.get_latest_scan()
        if scan is None:
            logger.warning("No scan data available to find nearest obstacle")
            return None
        
        min_distance = float('inf')
        nearest = None
        
        for angle, distance, quality in scan:
            if quality > 10 and distance < min_distance:
                min_distance = distance
                nearest = (angle, distance)
        
        return nearest
    
    def is_path_clear(self, angle, max_distance=1.0, angle_width=10):
        """Check if a path is clear in a specific direction
        
        Args:
            angle (float): Direction angle in degrees (0-359)
            max_distance (float): Maximum distance to check (meters)
            angle_width (float): Width of the cone to check (degrees)
        
        Returns:
            bool: True if path is clear, False if obstacles present
        """
        scan = self.get_latest_scan()
        if scan is None:
            logger.warning("No scan data available to check path")
            return False
        
        # Calculate angle range
        min_angle = (angle - angle_width/2) % 360
        max_angle = (angle + angle_width/2) % 360
        
        # Check if path is clear
        for point_angle, distance, quality in scan:
            # Check if the point is in the specified angle range
            in_range = False
            if min_angle <= max_angle:
                in_range = min_angle <= point_angle <= max_angle
            else:  # Angle range crosses 0/360
                in_range = point_angle >= min_angle or point_angle <= max_angle
            
            # If in range and distance is less than max_distance, path is not clear
            if in_range and distance < max_distance and quality > 10:
                return False
        
        return True

# Test function
def test_lidar_scanner():
    """Simple test function for the LIDAR scanner"""
    scanner = LidarScanner()
    if scanner.connect():
        print("Connected to LIDAR scanner")
        
        if scanner.start_scanning():
            print("Started scanning")
            
            # Wait for a few scans to be captured
            time.sleep(2)
            
            # Visualize a scan
            os.makedirs("../logs", exist_ok=True)
            scanner.visualize_scan("../logs/lidar_scan.png")
            print("Scan visualization saved to logs directory")
            
            # Check for obstacles
            obstacles = scanner.detect_obstacles()
            print(f"Detected {len(obstacles)} obstacles")
            
            # Get nearest obstacle
            nearest = scanner.get_nearest_obstacle()
            if nearest:
                angle, distance = nearest
                print(f"Nearest obstacle at angle {angle}° and distance {distance}m")
            
            # Stop scanning
            scanner.stop_scanning()
        
        scanner.disconnect()
        print("Test completed")
    else:
        print("Failed to connect to LIDAR scanner")

if __name__ == "__main__":
    # Set up logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_lidar_scanner() 