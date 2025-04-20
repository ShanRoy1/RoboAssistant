#!/usr/bin/env python3
# RoboAssistant - Remote Server
# Server that runs on the Raspberry Pi and communicates with the Windows controller

import os
import sys
import json
import time
import logging
import threading
import socket
import base64
import argparse
from pathlib import Path

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/server_{time.strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RoboAssistant.Server")

# Import hardware interfaces
try:
    sys.path.append('src')
    from robotic_arm import RoboticArm
    from depth_camera import DepthCamera
    from lidar_scanner import LidarScanner
    from neural_engine import NeuralEngine
    from voice_assistant import VoiceAssistant
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}")
    logger.error("Please make sure all dependencies are installed and src directory exists")
    sys.exit(1)

class RemoteServer:
    """Server that handles remote connections and commands"""
    
    def __init__(self, host='0.0.0.0', port=5000):
        """Initialize the server
        
        Args:
            host (str): Host to bind to
            port (int): Port to bind to
        """
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.clients = []
        
        # Initialize hardware components
        self.robotic_arm = None
        self.depth_camera = None
        self.lidar = None
        self.neural_engine = None
        self.voice_assistant = None
        
        # Current state
        self.current_mode = "standby"
        self.camera_streaming = False
        self.lidar_scanning = False
        
        # Load configuration
        self.config = self._load_config()
    
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
            # Default configuration
            return {
                "hardware": {
                    "robotic_arm": {"port": "/dev/ttyUSB0", "baudrate": 115200, "enabled": True},
                    "depth_camera": {"enabled": True},
                    "lidar": {"port": "/dev/ttyUSB1", "enabled": True},
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
                if not self.robotic_arm.connect():
                    logger.error("Failed to connect to robotic arm")
                    self.robotic_arm = None
            
            # Initialize depth camera if enabled
            if self.config["hardware"]["depth_camera"]["enabled"]:
                logger.info("Initializing Intel RealSense D455...")
                self.depth_camera = DepthCamera()
                if not self.depth_camera.connect():
                    logger.error("Failed to connect to depth camera")
                    self.depth_camera = None
            
            # Initialize LIDAR if enabled
            if self.config["hardware"]["lidar"]["enabled"]:
                logger.info("Initializing LIDAR A1M8...")
                port = self.config["hardware"]["lidar"]["port"]
                self.lidar = LidarScanner(port=port)
                if not self.lidar.connect():
                    logger.error("Failed to connect to LIDAR")
                    self.lidar = None
            
            # Initialize neural engine if enabled
            if self.config["hardware"]["coral_accelerator"]["enabled"]:
                logger.info("Initializing Neural Engine...")
                self.neural_engine = NeuralEngine()
                
                # If we have model paths defined, load them
                if "neural_engine" in self.config and "models" in self.config["neural_engine"]:
                    model_path = self.config["neural_engine"]["models"].get("object_detection")
                    label_path = self.config["neural_engine"]["labels"].get("object_detection")
                    if model_path and label_path:
                        if not self.neural_engine.load_model(model_path, label_path):
                            logger.error("Failed to load neural engine model")
                            self.neural_engine = None
            
            # Initialize voice assistant if we have any hardware to control
            if self.robotic_arm or self.depth_camera or self.lidar:
                logger.info("Initializing Voice Assistant...")
                self.voice_assistant = VoiceAssistant(
                    robot_arm=self.robotic_arm,
                    depth_camera=self.depth_camera,
                    lidar=self.lidar
                )
                if not self.voice_assistant.setup():
                    logger.error("Failed to set up voice assistant")
                    self.voice_assistant = None
            
            logger.info("Hardware initialization complete")
            return True
        except Exception as e:
            logger.error(f"Error initializing hardware: {str(e)}")
            return False
    
    def start(self):
        """Start the server"""
        if self.running:
            logger.warning("Server is already running")
            return False
        
        # Initialize hardware
        if not self.initialize_hardware():
            logger.error("Failed to initialize hardware")
            return False
        
        # Start server socket
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            
            self.running = True
            logger.info(f"Server started on {self.host}:{self.port}")
            
            # Start accepting connections
            self.accept_thread = threading.Thread(target=self._accept_connections)
            self.accept_thread.daemon = True
            self.accept_thread.start()
            
            return True
        except Exception as e:
            logger.error(f"Failed to start server: {str(e)}")
            return False
    
    def stop(self):
        """Stop the server"""
        if not self.running:
            logger.warning("Server is not running")
            return False
        
        logger.info("Stopping server")
        self.running = False
        
        # Close all client connections
        for client in self.clients:
            try:
                client['socket'].close()
            except:
                pass
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # Stop hardware components
        self._shutdown_hardware()
        
        logger.info("Server stopped")
        return True
    
    def _shutdown_hardware(self):
        """Shut down all hardware components"""
        logger.info("Shutting down hardware components")
        
        # Stop streaming if active
        if self.camera_streaming and self.depth_camera:
            self.depth_camera.stop_streaming()
            self.camera_streaming = False
        
        # Stop LIDAR if active
        if self.lidar_scanning and self.lidar:
            self.lidar.stop_scanning()
            self.lidar_scanning = False
        
        # Stop voice assistant if running
        if self.voice_assistant and hasattr(self.voice_assistant, 'running') and self.voice_assistant.running:
            self.voice_assistant.stop()
        
        # Disconnect hardware
        if self.depth_camera:
            self.depth_camera.disconnect()
        
        if self.lidar:
            self.lidar.disconnect()
        
        if self.robotic_arm:
            self.robotic_arm.disconnect()
    
    def _accept_connections(self):
        """Accept incoming connections"""
        logger.info("Waiting for connections")
        
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                logger.info(f"New connection from {address[0]}:{address[1]}")
                
                # Create a new client entry
                client = {
                    'socket': client_socket,
                    'address': address,
                    'thread': None
                }
                
                # Start a thread to handle this client
                client['thread'] = threading.Thread(
                    target=self._handle_client, 
                    args=(client,)
                )
                client['thread'].daemon = True
                client['thread'].start()
                
                # Add to client list
                self.clients.append(client)
                
            except Exception as e:
                if self.running:
                    logger.error(f"Error accepting connection: {str(e)}")
                    time.sleep(1)  # Avoid CPU spinning if there's an issue
    
    def _handle_client(self, client):
        """Handle communication with a client
        
        Args:
            client (dict): Client information including socket and address
        """
        client_socket = client['socket']
        address = client['address']
        
        logger.info(f"Handling client {address[0]}:{address[1]}")
        
        try:
            # Send initial status
            self._send_status(client_socket)
            
            # Handle commands from this client
            while self.running:
                # Receive data
                data = client_socket.recv(4096)
                if not data:
                    logger.info(f"Client {address[0]}:{address[1]} disconnected")
                    break
                
                # Parse command
                try:
                    command = json.loads(data.decode())
                    self._process_command(command, client_socket)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from client {address[0]}:{address[1]}")
                    response = {
                        "status": "error",
                        "message": "Invalid JSON format"
                    }
                    self._send_response(client_socket, response)
        
        except Exception as e:
            logger.error(f"Error handling client {address[0]}:{address[1]}: {str(e)}")
        
        finally:
            # Close connection and remove from clients list
            try:
                client_socket.close()
            except:
                pass
            
            try:
                self.clients.remove(client)
            except:
                pass
            
            logger.info(f"Client {address[0]}:{address[1]} handling complete")
    
    def _process_command(self, command, client_socket):
        """Process a command from a client
        
        Args:
            command (dict): Command to process
            client_socket (socket): Client socket to respond to
        """
        command_type = command.get('type')
        logger.info(f"Processing command: {command_type}")
        
        # Status request
        if command_type == 'status':
            self._send_status(client_socket)
        
        # Switch mode
        elif command_type == 'switch_mode':
            mode = command.get('mode')
            if mode in ['standby', 'assistant', 'scanner', 'surveillance']:
                self.current_mode = mode
                
                # Start or stop voice assistant based on mode
                if mode == 'assistant':
                    if self.voice_assistant and not getattr(self.voice_assistant, 'running', False):
                        self.voice_assistant.start()
                else:
                    if self.voice_assistant and getattr(self.voice_assistant, 'running', False):
                        self.voice_assistant.stop()
                
                response = {
                    "type": "switch_mode",
                    "status": "ok",
                    "mode": mode,
                    "message": f"Switched to {mode} mode"
                }
            else:
                response = {
                    "type": "switch_mode",
                    "status": "error",
                    "message": f"Invalid mode: {mode}"
                }
            
            self._send_response(client_socket, response)
        
        # Arm movement
        elif command_type == 'arm_move':
            direction = command.get('direction')
            if not self.robotic_arm:
                response = {
                    "type": "arm_move",
                    "status": "error",
                    "message": "Robotic arm is not available"
                }
            else:
                # Handle different movement directions
                success = False
                if direction == 'home':
                    success = self.robotic_arm.move_to_home()
                elif direction == 'left':
                    current = self.robotic_arm.joints["base"]
                    success = self.robotic_arm.move_joint("base", current + 20)
                elif direction == 'right':
                    current = self.robotic_arm.joints["base"]
                    success = self.robotic_arm.move_joint("base", current - 20)
                elif direction == 'up':
                    current = self.robotic_arm.joints["shoulder"]
                    success = self.robotic_arm.move_joint("shoulder", current - 20)
                elif direction == 'down':
                    current = self.robotic_arm.joints["shoulder"]
                    success = self.robotic_arm.move_joint("shoulder", current + 20)
                
                response = {
                    "type": "arm_move",
                    "status": "ok" if success else "error",
                    "message": f"Arm moved {direction}" if success else f"Failed to move arm {direction}"
                }
            
            self._send_response(client_socket, response)
        
        # Gripper control
        elif command_type == 'gripper':
            action = command.get('action')
            if not self.robotic_arm:
                response = {
                    "type": "gripper",
                    "status": "error",
                    "message": "Robotic arm is not available"
                }
            else:
                success = False
                if action == 'grab':
                    success = self.robotic_arm.grab_object()
                elif action == 'release':
                    success = self.robotic_arm.release_object()
                
                response = {
                    "type": "gripper",
                    "status": "ok" if success else "error",
                    "message": f"Gripper {action}" if success else f"Failed to {action} with gripper"
                }
            
            self._send_response(client_socket, response)
        
        # Gesture performance
        elif command_type == 'gesture':
            gesture = command.get('gesture')
            if not self.robotic_arm:
                response = {
                    "type": "gesture",
                    "status": "error",
                    "message": "Robotic arm is not available"
                }
            else:
                success = self.robotic_arm.perform_gesture(gesture)
                
                response = {
                    "type": "gesture",
                    "status": "ok" if success else "error",
                    "message": f"Performed gesture: {gesture}" if success else f"Failed to perform gesture: {gesture}"
                }
            
            self._send_response(client_socket, response)
        
        # Voice command
        elif command_type == 'voice_command':
            cmd = command.get('command')
            if not self.voice_assistant:
                response = {
                    "type": "voice_command",
                    "status": "error",
                    "message": "Voice assistant is not available",
                    "response": "Voice assistant is not available"
                }
            else:
                # Add the command to the voice assistant's queue
                self.voice_assistant.add_command_to_queue(cmd)
                
                # In a real implementation, we would need to capture the voice assistant's response
                # For now, we'll simulate a basic response
                simulated_response = f"I understood: {cmd}"
                
                response = {
                    "type": "voice_command",
                    "status": "ok",
                    "message": f"Processed command: {cmd}",
                    "response": simulated_response
                }
            
            self._send_response(client_socket, response)
        
        # Camera operations
        elif command_type == 'camera':
            action = command.get('action')
            if not self.depth_camera:
                response = {
                    "type": "camera",
                    "status": "error",
                    "message": "Depth camera is not available"
                }
            else:
                if action == 'start':
                    success = self.depth_camera.start_streaming()
                    self.camera_streaming = success
                    response = {
                        "type": "camera",
                        "status": "ok" if success else "error",
                        "message": "Camera streaming started" if success else "Failed to start camera streaming"
                    }
                
                elif action == 'stop':
                    success = self.depth_camera.stop_streaming()
                    self.camera_streaming = False
                    response = {
                        "type": "camera",
                        "status": "ok" if success else "error",
                        "message": "Camera streaming stopped" if success else "Failed to stop camera streaming"
                    }
                
                elif action == 'snapshot':
                    # Take a snapshot
                    if not self.camera_streaming:
                        self.depth_camera.start_streaming()
                        time.sleep(1)  # Let the camera stabilize
                    
                    color, depth = self.depth_camera.get_frames()
                    
                    if color is not None:
                        import cv2
                        import numpy as np
                        
                        # Save snapshot
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        os.makedirs("snapshots", exist_ok=True)
                        filename = f"snapshots/snapshot_{timestamp}.jpg"
                        cv2.imwrite(filename, color)
                        
                        response = {
                            "type": "camera",
                            "status": "ok",
                            "action": "snapshot",
                            "message": f"Snapshot saved as {filename}"
                        }
                    else:
                        response = {
                            "type": "camera",
                            "status": "error",
                            "message": "Failed to capture snapshot"
                        }
                
                elif action == 'detect':
                    # Object detection
                    if not self.neural_engine:
                        response = {
                            "type": "camera",
                            "status": "error",
                            "message": "Neural engine is not available"
                        }
                    else:
                        if not self.camera_streaming:
                            self.depth_camera.start_streaming()
                            time.sleep(1)
                        
                        color, _ = self.depth_camera.get_frames()
                        
                        if color is not None:
                            # Detect objects
                            detections = self.neural_engine.detect_objects(color)
                            
                            # Draw detections on image
                            result_image = self.neural_engine.draw_detections(color, detections)
                            
                            # Save image with detections
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            os.makedirs("detections", exist_ok=True)
                            filename = f"detections/detection_{timestamp}.jpg"
                            cv2.imwrite(filename, result_image)
                            
                            # Create response with detections
                            objects = []
                            for det in detections:
                                objects.append({
                                    "label": det.get("label", "unknown"),
                                    "confidence": det.get("score", 0.0),
                                    "bbox": det.get("bbox", [0, 0, 0, 0])
                                })
                            
                            response = {
                                "type": "camera",
                                "status": "ok",
                                "action": "detect",
                                "message": f"Detected {len(detections)} objects",
                                "objects": objects,
                                "image_path": filename
                            }
                        else:
                            response = {
                                "type": "camera",
                                "status": "error",
                                "message": "Failed to capture image for detection"
                            }
                
                else:
                    response = {
                        "type": "camera",
                        "status": "error",
                        "message": f"Unknown camera action: {action}"
                    }
            
            self._send_response(client_socket, response)
        
        # LIDAR operations
        elif command_type == 'lidar':
            action = command.get('action')
            if not self.lidar:
                response = {
                    "type": "lidar",
                    "status": "error",
                    "message": "LIDAR scanner is not available"
                }
            else:
                if action == 'start':
                    success = self.lidar.start_scanning()
                    self.lidar_scanning = success
                    response = {
                        "type": "lidar",
                        "status": "ok" if success else "error",
                        "message": "LIDAR scanning started" if success else "Failed to start LIDAR scanning"
                    }
                
                elif action == 'stop':
                    success = self.lidar.stop_scanning()
                    self.lidar_scanning = False
                    response = {
                        "type": "lidar",
                        "status": "ok" if success else "error",
                        "message": "LIDAR scanning stopped" if success else "Failed to stop LIDAR scanning"
                    }
                
                elif action == 'create_map':
                    # Create a map visualization
                    if not self.lidar_scanning:
                        self.lidar.start_scanning()
                        time.sleep(2)  # Let the LIDAR collect data
                    
                    # Generate and save visualization
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    os.makedirs("maps", exist_ok=True)
                    filename = f"maps/lidar_map_{timestamp}.png"
                    
                    success = self.lidar.visualize_scan(filename)
                    
                    response = {
                        "type": "lidar",
                        "status": "ok" if success else "error",
                        "message": f"Map created at {filename}" if success else "Failed to create map"
                    }
                
                elif action == 'detect_obstacles':
                    # Detect obstacles in LIDAR data
                    if not self.lidar_scanning:
                        self.lidar.start_scanning()
                        time.sleep(1)
                    
                    obstacles = self.lidar.detect_obstacles()
                    
                    response = {
                        "type": "lidar",
                        "status": "ok",
                        "message": f"Detected {len(obstacles)} obstacles",
                        "obstacles": obstacles
                    }
                
                elif action == 'check_path':
                    # Check if path is clear
                    angle = command.get('angle', 0)
                    distance = command.get('distance', 2.0)
                    
                    if not self.lidar_scanning:
                        self.lidar.start_scanning()
                        time.sleep(1)
                    
                    is_clear = self.lidar.is_path_clear(angle, distance)
                    
                    response = {
                        "type": "lidar",
                        "status": "ok",
                        "message": "Path is clear" if is_clear else "Path is blocked",
                        "is_clear": is_clear
                    }
                
                else:
                    response = {
                        "type": "lidar",
                        "status": "error",
                        "message": f"Unknown LIDAR action: {action}"
                    }
            
            self._send_response(client_socket, response)
        
        # Unknown command type
        else:
            response = {
                "type": "unknown",
                "status": "error",
                "message": f"Unknown command type: {command_type}"
            }
            self._send_response(client_socket, response)
    
    def _send_status(self, client_socket):
        """Send current status to a client
        
        Args:
            client_socket (socket): Client socket to send status to
        """
        status = {
            "type": "status",
            "status": "ok",
            "mode": self.current_mode,
            "devices": {
                "robotic_arm": self.robotic_arm is not None,
                "depth_camera": self.depth_camera is not None,
                "lidar": self.lidar is not None,
                "neural_engine": self.neural_engine is not None,
                "voice_assistant": self.voice_assistant is not None
            },
            "streaming": {
                "camera": self.camera_streaming,
                "lidar": self.lidar_scanning
            }
        }
        
        self._send_response(client_socket, status)
    
    def _send_response(self, client_socket, response):
        """Send a response to a client
        
        Args:
            client_socket (socket): Client socket to send to
            response (dict): Response to send
        """
        try:
            # Add timestamp to the response
            response["timestamp"] = time.time()
            
            # Send the JSON response
            client_socket.sendall(json.dumps(response).encode())
        except Exception as e:
            logger.error(f"Error sending response: {str(e)}")

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='RoboAssistant Remote Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    return parser.parse_args()

def main():
    """Main entry point"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create and start the server
    server = RemoteServer(host=args.host, port=args.port)
    
    if server.start():
        logger.info("Server started successfully")
        
        # Wait for Ctrl+C
        try:
            print("Press Ctrl+C to stop the server")
            while server.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping server")
        finally:
            server.stop()
    else:
        logger.error("Failed to start server")

if __name__ == "__main__":
    main() 