#!/usr/bin/env python3
# RoboAssistant - Main Application
# Integrates Waveshare Robotic Arm, Intel RealSense D455, LIDAR A1M8, and Coral USB Accelerator

import os
import sys
import time
import logging
import threading
import json
import argparse
from datetime import datetime
from pathlib import Path

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RoboAssistant")

# Import hardware interfaces
try:
    from robotic_arm import RoboticArm
    from depth_camera import DepthCamera
    from lidar_scanner import LidarScanner
    from neural_engine import NeuralEngine
    from voice_assistant import VoiceAssistant
    from fusion_qc import FusionQC
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    logger.error("Please make sure all dependencies are installed (pip install -r requirements.txt)")
    sys.exit(1)

class RoboAssistant:
    """Main class for the Robotic Arm Assistant application"""
    
    def __init__(self):
        logger.info("Initializing RoboAssistant")
        self.mode = "standby"  # Possible modes: standby, assistant, scanner, surveillance, qc
        
        # Load configuration
        self.config = self._load_config()
        
        # Hardware components (will be initialized later)
        self.robotic_arm = None
        self.depth_camera = None
        self.lidar = None
        self.neural_engine = None
        self.voice_assistant = None
        self.fusion_qc = None
        
        # Status flags
        self.running = False
        self.initialized = False
    
    def _load_config(self):
        """Load configuration from settings.json"""
        try:
            config_path = Path('config') / 'settings.json'
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            logger.info("Using default configuration")
            return {
                "hardware": {
                    "robotic_arm": {"port": "COM3", "baudrate": 115200, "enabled": True},
                    "depth_camera": {"enabled": True},
                    "lidar": {"port": "COM4", "enabled": True},
                    "coral_accelerator": {"enabled": True}
                },
                "application": {
                    "default_mode": "assistant",
                    "log_level": "INFO"
                }
            }
    
    def initialize_hardware(self):
        """Initialize all hardware components"""
        logger.info("Initializing hardware components")
        try:
            # Initialize robotic arm if enabled
            if self.config["hardware"]["robotic_arm"]["enabled"]:
                logger.info("Initializing Waveshare Robotic Arm...")
                port = self.config["hardware"]["robotic_arm"]["port"]
                baudrate = self.config["hardware"]["robotic_arm"]["baudrate"]
                self.robotic_arm = RoboticArm(port=port, baudrate=baudrate)
                self.robotic_arm.connect()
            
            # Initialize depth camera if enabled
            if self.config["hardware"]["depth_camera"]["enabled"]:
                logger.info("Initializing Intel RealSense D455...")
                self.depth_camera = DepthCamera()
                self.depth_camera.connect()
            
            # Initialize LIDAR if enabled
            if self.config["hardware"]["lidar"]["enabled"]:
                logger.info("Initializing LIDAR A1M8...")
                port = self.config["hardware"]["lidar"]["port"]
                self.lidar = LidarScanner(port=port)
                self.lidar.connect()
            
            # Initialize neural engine if enabled
            if self.config["hardware"]["coral_accelerator"]["enabled"]:
                logger.info("Initializing Neural Engine...")
                self.neural_engine = NeuralEngine()
                
                # If we have model paths defined, load them
                if "neural_engine" in self.config and "models" in self.config["neural_engine"]:
                    model_path = self.config["neural_engine"]["models"].get("object_detection")
                    label_path = self.config["neural_engine"]["labels"].get("object_detection")
                    if model_path and label_path:
                        self.neural_engine.load_model(model_path, label_path)
            
            # Initialize voice assistant if we have any hardware to control
            if self.robotic_arm or self.depth_camera or self.lidar:
                logger.info("Initializing Voice Assistant...")
                self.voice_assistant = VoiceAssistant(
                    robot_arm=self.robotic_arm,
                    depth_camera=self.depth_camera,
                    lidar=self.lidar
                )
                self.voice_assistant.setup()
            
            # Initialize Fusion QC system if both depth camera and LIDAR are available
            if self.depth_camera and self.lidar:
                logger.info("Initializing Fusion QC system...")
                self.fusion_qc = FusionQC(depth_camera=self.depth_camera, lidar_scanner=self.lidar)
                
                # Try to load existing calibration
                calibration_file = Path('config') / 'sensor_calibration.json'
                if calibration_file.exists():
                    self.fusion_qc.load_calibration(str(calibration_file))
            
            self.initialized = True
            logger.info("All hardware components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize hardware: {str(e)}")
            return False
    
    def start(self, initial_mode=None):
        """Start the RoboAssistant application
        
        Args:
            initial_mode (str, optional): Initial operating mode
        
        Returns:
            bool: Success status
        """
        if not self.initialized:
            success = self.initialize_hardware()
            if not success:
                logger.error("Could not start RoboAssistant due to initialization failure")
                return False
        
        logger.info("Starting RoboAssistant")
        self.running = True
        
        # Set initial mode
        if initial_mode:
            self.mode = initial_mode
        else:
            # Use default mode from config
            self.mode = self.config["application"].get("default_mode", "assistant")
        
        logger.info(f"Initial mode set to: {self.mode}")
        return True
    
    def stop(self):
        """Stop the RoboAssistant application"""
        logger.info("Stopping RoboAssistant")
        self.running = False
        
        # Stop voice assistant if running
        if self.voice_assistant and hasattr(self.voice_assistant, 'running') and self.voice_assistant.running:
            self.voice_assistant.stop()
        
        # Stop hardware components
        if self.depth_camera:
            self.depth_camera.disconnect()
        
        if self.lidar:
            self.lidar.disconnect()
        
        if self.robotic_arm:
            self.robotic_arm.disconnect()
        
        logger.info("RoboAssistant stopped")
        return True
    
    def switch_mode(self, new_mode):
        """Switch between operating modes"""
        if new_mode not in ["standby", "assistant", "scanner", "surveillance", "qc"]:
            logger.error(f"Invalid mode: {new_mode}")
            return False
        
        logger.info(f"Switching mode from {self.mode} to {new_mode}")
        self.mode = new_mode
        
        # Perform mode-specific initialization
        if new_mode == "assistant":
            # Start voice assistant if not already running
            if self.voice_assistant and not getattr(self.voice_assistant, 'running', False):
                self.voice_assistant.start()
        elif new_mode == "scanner":
            # Stop voice assistant if running
            if self.voice_assistant and getattr(self.voice_assistant, 'running', False):
                self.voice_assistant.stop()
        elif new_mode == "surveillance":
            # Stop voice assistant if running
            if self.voice_assistant and getattr(self.voice_assistant, 'running', False):
                self.voice_assistant.stop()
        elif new_mode == "qc":
            # Stop voice assistant if running
            if self.voice_assistant and getattr(self.voice_assistant, 'running', False):
                self.voice_assistant.stop()
        
        return True
    
    def run_assistant_mode(self):
        """Run in Jarvis-like assistant mode - listening for commands and manipulating objects"""
        logger.info("Running in assistant mode")
        
        if not self.voice_assistant:
            logger.error("Cannot run assistant mode: Voice assistant not initialized")
            print("Error: Voice assistant not initialized. Make sure hardware is connected.")
            return
        
        if not getattr(self.voice_assistant, 'running', False):
            self.voice_assistant.start()
        
        print("\nAssistant Mode (Jarvis)")
        print("------------------------")
        print("The voice assistant is active and listening for commands.")
        print("You can say things like:")
        print("  - 'wave hello'")
        print("  - 'scan the area'")
        print("  - 'what is that object'")
        print("  - 'move to the left'")
        print("Enter 'back' to return to the main menu.\n")
        
        # Wait for user input to return to main menu
        while self.running and self.mode == "assistant":
            command = input("> ")
            if command.lower() == 'back':
                break
            else:
                # Process command through voice assistant
                self.voice_assistant.add_command_to_queue(command)
    
    def run_scanner_mode(self):
        """Run in 3D scanning mode - creating mesh from Intel camera and LIDAR data"""
        logger.info("Running in scanner mode")
        
        if not self.depth_camera and not self.lidar:
            logger.error("Cannot run scanner mode: No scanning hardware initialized")
            print("Error: No scanning hardware initialized. Make sure the depth camera or LIDAR is connected.")
            return
        
        print("\nScanner Mode (3D Scanning)")
        print("--------------------------")
        print("This mode will create a 3D scan by combining depth camera and LIDAR data.")
        print("Options:")
        print("  1. Quick Scan (5 seconds)")
        print("  2. Standard Scan (15 seconds)")
        print("  3. Detailed Scan (30 seconds)")
        print("  4. QC Scan (Compare with Fusion 360 Model)")
        print("  5. Calibrate Sensors")
        print("  6. Back to main menu")
        
        choice = input("Select an option (1-6): ")
        
        if choice == '6':
            return
        
        # Calibrate sensors
        if choice == '5':
            if not self.fusion_qc:
                print("Error: Fusion QC system not initialized. Make sure both depth camera and LIDAR are connected.")
                input("Press Enter to continue...")
                return
                
            print("\nCalibrating sensors...")
            print("Please place a checkerboard pattern or object visible to both sensors.")
            print("Keep the area still during calibration.")
            
            success = self.fusion_qc.calibrate_sensors()
            
            if success:
                print("\nCalibration completed successfully!")
            else:
                print("\nCalibration failed. Please try again with better lighting and positioning.")
                
            input("Press Enter to continue...")
            return
            
        # QC Scan
        if choice == '4':
            if not self.fusion_qc:
                print("Error: Fusion QC system not initialized. Make sure both depth camera and LIDAR are connected.")
                input("Press Enter to continue...")
                return
                
            # Ask for the reference model file
            print("\nQC Scan - Compare with Fusion 360 Model")
            model_file = input("Enter path to reference model file (STL, OBJ): ")
            
            if not os.path.exists(model_file):
                print(f"Error: File {model_file} not found.")
                input("Press Enter to continue...")
                return
                
            # Load the reference model
            print(f"\nLoading reference model from {model_file}...")
            if not self.fusion_qc.load_fusion360_model(model_file):
                print("Error: Failed to load reference model.")
                input("Press Enter to continue...")
                return
                
            # Create combined scan
            print("\nCreating high-quality combined scan (30s)...")
            print("Please keep the object still during scanning.")
            
            if not self.fusion_qc.create_combined_scan(scan_duration=30, high_quality=True):
                print("Error: Failed to create scan.")
                input("Press Enter to continue...")
                return
                
            # Align scan to model
            print("\nAligning scan to reference model...")
            transform = self.fusion_qc.align_scan_to_model()
            
            if transform is None:
                print("Error: Failed to align scan to model.")
                input("Press Enter to continue...")
                return
                
            # Compare with reference model
            print("\nComparing scan to reference model...")
            result = self.fusion_qc.compare_with_model()
            
            if result:
                # Display results
                print("\n===== QC Results =====")
                print(f"Status: {result['status']}")
                print(f"Mean deviation: {result['mean_distance']*1000:.2f}mm")
                print(f"Maximum deviation: {result['max_distance']*1000:.2f}mm")
                print(f"Standard deviation: {result['std_distance']*1000:.2f}mm")
                print(f"Points outside tolerance: {result['percentage_outside']:.2f}% ({result['points_outside_tolerance']} of {result['total_points']})")
                print(f"Tolerance: {result['tolerance']*1000:.2f}mm")
                print(f"Report saved to: output/reports/qc_report_{time.strftime('%Y%m%d_%H%M%S')}.json")
                print(f"Visualization saved to: output/visualizations/")
            else:
                print("Error: Failed to compare scan with model.")
                
            input("\nPress Enter to continue...")
            return
        
        # Determine scan duration
        if choice == '1':
            duration = 5
            quality = "Quick"
        elif choice == '2':
            duration = 15
            quality = "Standard"
        elif choice == '3':
            duration = 30
            quality = "Detailed"
        else:
            print("Invalid choice, using Standard scan.")
            duration = 15
            quality = "Standard"
        
        # Start the scanning process
        print(f"\nStarting {quality} scan ({duration} seconds)...")
        print("Please keep the area still during scanning.")
        
        # Start depth camera if available
        depth_data = None
        if self.depth_camera:
            self.depth_camera.start_streaming()
        
        # Start LIDAR if available
        lidar_data = None
        if self.lidar:
            self.lidar.start_scanning()
        
        # Simulate scanning progress
        for i in range(duration):
            progress = (i + 1) / duration * 100
            sys.stdout.write(f"\rScanning: {progress:.1f}% complete...")
            sys.stdout.flush()
            time.sleep(1)
        
        # Collect data from sensors
        if self.depth_camera:
            color, depth = self.depth_camera.get_frames()
            if depth is not None:
                # Save depth data visualization
                os.makedirs("logs", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"logs/depth_scan_{timestamp}.jpg"
                # In a real implementation, this would save actual data
                # For now, save a visualization
                import cv2
                import numpy as np
                depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite(filename, depth_vis)
                print(f"\nDepth data saved to {filename}")
                
                # Save color image
                color_filename = f"logs/color_scan_{timestamp}.jpg"
                cv2.imwrite(color_filename, color)
                print(f"Color image saved to {color_filename}")
                
                # Save point cloud if available
                pc_filename = f"logs/pointcloud_{timestamp}.ply"
                self.depth_camera.save_point_cloud(pc_filename)
                print(f"Point cloud saved to {pc_filename}")
            
            # Stop depth camera
            self.depth_camera.stop_streaming()
        
        if self.lidar:
            # Save LIDAR scan visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            lidar_filename = f"logs/lidar_scan_{timestamp}.png"
            self.lidar.visualize_scan(lidar_filename)
            print(f"LIDAR scan saved to {lidar_filename}")
            
            # Stop LIDAR
            self.lidar.stop_scanning()
        
        print("\nScan completed!")
        if not self.depth_camera and not self.lidar:
            print("No scan data was collected because no scanning hardware was available.")
        
        input("Press Enter to continue...")
    
    def run_surveillance_mode(self):
        """Run in surveillance mode - scanning area and detecting changes"""
        logger.info("Running in surveillance mode")
        
        if not self.lidar and not self.depth_camera:
            logger.error("Cannot run surveillance mode: No sensing hardware initialized")
            print("Error: No sensing hardware initialized. Make sure the depth camera or LIDAR is connected.")
            return
        
        print("\nSurveillance Mode (Area Monitoring)")
        print("----------------------------------")
        print("This mode will monitor the area and detect changes.")
        print("Options:")
        print("  1. Create baseline scan")
        print("  2. Start monitoring")
        print("  3. Back to main menu")
        
        choice = input("Select an option (1-3): ")
        
        if choice == '3':
            return
        
        if choice == '1':
            # Create baseline scan
            print("\nCreating baseline scan...")
            print("Please keep the area still during scanning.")
            
            # Start sensors
            if self.depth_camera:
                self.depth_camera.start_streaming()
                time.sleep(2)  # Allow time for streaming to start
            
            if self.lidar:
                self.lidar.start_scanning()
                time.sleep(2)  # Allow time for scanning to start
            
            # Simulate creating baseline
            for i in range(5):
                progress = (i + 1) / 5 * 100
                sys.stdout.write(f"\rCreating baseline: {progress:.1f}% complete...")
                sys.stdout.flush()
                time.sleep(1)
            
            # Save baseline data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.depth_camera:
                color, depth = self.depth_camera.get_frames()
                if depth is not None:
                    # Save baseline depth image
                    baseline_filename = f"logs/baseline_depth_{timestamp}.npy"
                    if not os.path.exists("logs"):
                        os.makedirs("logs")
                    import numpy as np
                    np.save(baseline_filename, depth)
                    print(f"\nBaseline depth data saved to {baseline_filename}")
                
                self.depth_camera.stop_streaming()
            
            if self.lidar:
                # Save baseline LIDAR scan
                baseline_lidar = f"logs/baseline_lidar_{timestamp}.data"
                scan = self.lidar.get_latest_scan()
                if scan:
                    # In a real implementation, this would save the actual scan data
                    # For now, just note that it would be saved
                    print(f"Baseline LIDAR scan would be saved to {baseline_lidar}")
                
                self.lidar.stop_scanning()
            
            print("\nBaseline scan completed!")
            input("Press Enter to continue...")
            
        elif choice == '2':
            # Start monitoring
            print("\nStarting area monitoring...")
            print("The system will alert you if changes are detected.")
            print("Press Enter at any time to stop monitoring.")
            
            # Start sensors
            if self.depth_camera:
                self.depth_camera.start_streaming()
            
            if self.lidar:
                self.lidar.start_scanning()
            
            # Load baseline if available (in a real implementation)
            baseline_depth = None
            baseline_lidar = None
            
            # Simulate monitoring with occasional alerts
            monitoring = True
            start_time = time.time()
            alert_times = [10, 25, 40]  # Times to simulate alerts (in seconds)
            
            import threading
            stop_event = threading.Event()
            
            def check_input():
                input("Press Enter to stop monitoring...\n")
                stop_event.set()
            
            input_thread = threading.Thread(target=check_input)
            input_thread.daemon = True
            input_thread.start()
            
            while monitoring and not stop_event.is_set():
                elapsed = time.time() - start_time
                
                # Simulate alerts at specific times
                if any(abs(elapsed - t) < 0.5 for t in alert_times):
                    if self.depth_camera:
                        print("\n🚨 ALERT: Movement detected in depth camera view!")
                    elif self.lidar:
                        print("\n🚨 ALERT: New obstacle detected by LIDAR!")
                    
                    time.sleep(1)  # Pause to make the alert noticeable
                
                # Update status every second
                sys.stdout.write(f"\rMonitoring for {int(elapsed)} seconds...")
                sys.stdout.flush()
                time.sleep(0.1)
                
                # In a real implementation, would check for actual changes here
                
                # Stop after 60 seconds for the demo
                if elapsed > 60:
                    print("\nDemo completed after 60 seconds.")
                    monitoring = False
            
            # Stop sensors
            if self.depth_camera:
                self.depth_camera.stop_streaming()
            
            if self.lidar:
                self.lidar.stop_scanning()
            
            print("\nMonitoring stopped.")
            input("Press Enter to continue...")

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='RoboAssistant Application')
    parser.add_argument('--mode', choices=['assistant', 'scanner', 'surveillance', 'qc'],
                        help='Initial operating mode')
    parser.add_argument('--server', action='store_true', help='Start in server mode for remote control')
    return parser.parse_args()

def main():
    """Main entry point for the application"""
    # Parse command-line arguments
    args = parse_arguments()
    
    logger.info("Starting RoboAssistant application")
    
    # Create and start the application
    app = RoboAssistant()
    if not app.start(args.mode):
        logger.error("Failed to start application")
        return
    
    try:
        # Simple command line interface
        while app.running:
            print("\nRoboAssistant - Current Mode:", app.mode)
            print("1. Assistant Mode (Jarvis)")
            print("2. Scanner Mode (3D Scanning)")
            print("3. Surveillance Mode (Area Monitoring)")
            print("4. QC Mode (Quality Control)")
            print("0. Exit")
            
            choice = input("Select mode (0-4): ")
            
            if choice == "1":
                app.switch_mode("assistant")
                app.run_assistant_mode()
            elif choice == "2":
                app.switch_mode("scanner")
                app.run_scanner_mode()
            elif choice == "3":
                app.switch_mode("surveillance")
                app.run_surveillance_mode()
            elif choice == "4":
                if app.fusion_qc:
                    app.switch_mode("qc")
                    app.run_scanner_mode()
                else:
                    print("Error: QC mode requires both depth camera and LIDAR to be connected.")
            elif choice == "0":
                app.stop()
                break
            else:
                print("Invalid choice, please try again.")
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
        app.stop()
    
    logger.info("RoboAssistant application terminated")

if __name__ == "__main__":
    main() 