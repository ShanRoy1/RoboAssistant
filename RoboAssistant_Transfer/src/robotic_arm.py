#!/usr/bin/env python3
# RoboAssistant - Robotic Arm Module
# Interface for the Waveshare Robotic Arm (RoArm M2-S)

import time
import logging
import serial
import threading

logger = logging.getLogger("RoboAssistant.RoboticArm")

class RoboticArm:
    """Interface for controlling the Waveshare Robotic Arm"""
    
    def __init__(self, port="COM3", baudrate=115200):
        """Initialize the robotic arm connection
        
        Args:
            port (str): Serial port for the robotic arm
            baudrate (int): Baud rate for serial communication
        """
        logger.info(f"Initializing robotic arm on port {port} with baudrate {baudrate}")
        
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.connected = False
        
        # Arm joint positions (will be calibrated on startup)
        self.joints = {
            "base": 90,      # Base rotation (0-180)
            "shoulder": 90,  # Shoulder joint (0-180)
            "elbow": 90,     # Elbow joint (0-180)
            "wrist": 90,     # Wrist angle (0-180)
            "gripper": 90    # Gripper (0=open, 180=closed)
        }
        
        # Movement limits to avoid self-collision
        self.limits = {
            "base": (0, 180),
            "shoulder": (15, 165),
            "elbow": (0, 180),
            "wrist": (0, 180),
            "gripper": (0, 180)
        }
        
        # Connection and initialization
        self.lock = threading.Lock()
    
    def connect(self):
        """Establish connection to the robotic arm"""
        try:
            logger.info(f"Connecting to robotic arm on {self.port}")
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            time.sleep(2)  # Allow time for the connection to establish
            self.connected = True
            logger.info("Successfully connected to robotic arm")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to robotic arm: {str(e)}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the robotic arm"""
        if self.serial_connection and self.connected:
            logger.info("Disconnecting from robotic arm")
            self.move_to_home()  # Return to home position before disconnecting
            self.serial_connection.close()
            self.connected = False
            logger.info("Disconnected from robotic arm")
            return True
        return False
    
    def send_command(self, command):
        """Send a command to the robotic arm
        
        Args:
            command (str): Command string to send to the arm
        
        Returns:
            bool: Success status
        """
        if not self.connected:
            logger.error("Cannot send command: Not connected to robotic arm")
            return False
        
        try:
            with self.lock:
                logger.debug(f"Sending command: {command}")
                self.serial_connection.write(f"{command}\n".encode())
                time.sleep(0.1)  # Small delay to ensure command is processed
                return True
        except Exception as e:
            logger.error(f"Failed to send command: {str(e)}")
            return False
    
    def move_joint(self, joint_name, angle, speed=50):
        """Move a specific joint to the target angle
        
        Args:
            joint_name (str): Name of the joint to move
            angle (int): Target angle (0-180)
            speed (int): Movement speed (1-100)
        
        Returns:
            bool: Success status
        """
        if joint_name not in self.joints:
            logger.error(f"Invalid joint name: {joint_name}")
            return False
        
        # Ensure angle is within limits
        min_angle, max_angle = self.limits[joint_name]
        if angle < min_angle or angle > max_angle:
            logger.warning(f"Angle {angle} for {joint_name} outside limits ({min_angle}-{max_angle})")
            angle = max(min_angle, min(angle, max_angle))
        
        # Construct command (format will depend on specific arm protocol)
        # This is a placeholder and should be replaced with actual command structure
        command = f"MOVE {joint_name.upper()} {angle} {speed}"
        
        success = self.send_command(command)
        if success:
            self.joints[joint_name] = angle
            logger.info(f"Moved {joint_name} to {angle} degrees")
        return success
    
    def move_to_position(self, base=None, shoulder=None, elbow=None, wrist=None, gripper=None, speed=50):
        """Move the arm to a specific position
        
        Args:
            base (int, optional): Base rotation angle
            shoulder (int, optional): Shoulder angle
            elbow (int, optional): Elbow angle
            wrist (int, optional): Wrist angle
            gripper (int, optional): Gripper angle
            speed (int): Movement speed (1-100)
        
        Returns:
            bool: Success status
        """
        success = True
        
        # Move each specified joint
        if base is not None:
            success = success and self.move_joint("base", base, speed)
        
        if shoulder is not None:
            success = success and self.move_joint("shoulder", shoulder, speed)
        
        if elbow is not None:
            success = success and self.move_joint("elbow", elbow, speed)
        
        if wrist is not None:
            success = success and self.move_joint("wrist", wrist, speed)
        
        if gripper is not None:
            success = success and self.move_joint("gripper", gripper, speed)
        
        return success
    
    def move_to_home(self):
        """Move the arm to the home position"""
        logger.info("Moving arm to home position")
        return self.move_to_position(
            base=90, 
            shoulder=90, 
            elbow=90, 
            wrist=90, 
            gripper=90
        )
    
    def grab_object(self):
        """Close the gripper to grab an object"""
        logger.info("Closing gripper to grab object")
        return self.move_joint("gripper", 180)  # Close gripper
    
    def release_object(self):
        """Open the gripper to release an object"""
        logger.info("Opening gripper to release object")
        return self.move_joint("gripper", 0)  # Open gripper
    
    def perform_gesture(self, gesture_name):
        """Perform a predefined gesture
        
        Args:
            gesture_name (str): Name of the gesture to perform
        
        Returns:
            bool: Success status
        """
        gestures = {
            "wave": self._gesture_wave,
            "nod": self._gesture_nod,
            "point": self._gesture_point,
            "pick": self._gesture_pick_and_place
        }
        
        if gesture_name not in gestures:
            logger.error(f"Unknown gesture: {gesture_name}")
            return False
        
        logger.info(f"Performing gesture: {gesture_name}")
        gestures[gesture_name]()
        return True
    
    def _gesture_wave(self):
        """Wave gesture animation"""
        self.move_to_position(base=90, shoulder=45, elbow=45, wrist=90)
        for _ in range(3):
            self.move_joint("wrist", 45)
            time.sleep(0.5)
            self.move_joint("wrist", 135)
            time.sleep(0.5)
        self.move_to_home()
    
    def _gesture_nod(self):
        """Nodding gesture animation"""
        self.move_to_position(base=90, shoulder=90, elbow=90)
        for _ in range(3):
            self.move_joint("wrist", 45)
            time.sleep(0.5)
            self.move_joint("wrist", 90)
            time.sleep(0.5)
        self.move_to_home()
    
    def _gesture_point(self):
        """Pointing gesture animation"""
        self.move_to_position(base=90, shoulder=45, elbow=90, wrist=0)
        time.sleep(1)
        self.move_to_home()
    
    def _gesture_pick_and_place(self):
        """Pick and place demonstration gesture"""
        # Move to object position
        self.move_to_position(base=45, shoulder=45, elbow=45, wrist=90, gripper=0)
        time.sleep(1)
        
        # Grab object
        self.grab_object()
        time.sleep(1)
        
        # Move to placement position
        self.move_to_position(base=135, shoulder=45, elbow=45, wrist=90)
        time.sleep(1)
        
        # Release object
        self.release_object()
        time.sleep(1)
        
        # Return to home
        self.move_to_home()

# Test function
def test_robotic_arm():
    """Simple test function for the robotic arm"""
    arm = RoboticArm()
    if arm.connect():
        print("Connected to robotic arm")
        arm.move_to_home()
        arm.perform_gesture("wave")
        arm.disconnect()
        print("Test completed")
    else:
        print("Failed to connect to robotic arm")

if __name__ == "__main__":
    # Set up logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_robotic_arm() 