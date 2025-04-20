#!/usr/bin/env python3
# RoboAssistant - Voice Assistant Module
# Provides Jarvis-like voice interaction capabilities

import time
import logging
import threading
import queue
import re
import random
import os
import math

# In a real implementation, these would be the actual imports
# import speech_recognition as sr
# import pyttsx3
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords

logger = logging.getLogger("RoboAssistant.VoiceAssistant")

class VoiceAssistant:
    """Voice assistant for Jarvis-like interaction with the robotic arm"""
    
    def __init__(self, robot_arm=None, depth_camera=None, lidar=None):
        """Initialize the voice assistant
        
        Args:
            robot_arm: RoboticArm instance
            depth_camera: DepthCamera instance
            lidar: LidarScanner instance
        """
        logger.info("Initializing Voice Assistant")
        
        # Connected hardware
        self.robot_arm = robot_arm
        self.depth_camera = depth_camera
        self.lidar = lidar
        
        # Speech recognition and synthesis components
        self.recognizer = None
        self.engine = None
        self.microphone = None
        
        # Command processing
        self.running = False
        self.command_queue = queue.Queue()
        self.processing_thread = None
        self.listening_thread = None
        
        # Wake word/phrase
        self.wake_phrase = "hey jarvis"
        
        # Command patterns and responses
        self.command_patterns = self._initialize_command_patterns()
    
    def setup(self):
        """Set up the voice recognition and synthesis components"""
        logger.info("Setting up voice recognition and synthesis")
        
        try:
            # In a real implementation, this would initialize the actual components
            # self.recognizer = sr.Recognizer()
            # self.engine = pyttsx3.init()
            # self.engine.setProperty('rate', 180)  # Speed of speech
            # self.engine.setProperty('volume', 0.9)  # Volume (0-1)
            # voices = self.engine.getProperty('voices')
            # self.engine.setProperty('voice', voices[0].id)  # Voice type
            
            # For testing, we'll simulate a successful setup
            logger.info("Voice assistant setup complete")
            return True
        except Exception as e:
            logger.error(f"Failed to set up voice assistant: {str(e)}")
            return False
    
    def start(self):
        """Start the voice assistant"""
        if self.running:
            logger.warning("Voice assistant is already running")
            return False
        
        logger.info("Starting voice assistant")
        self.running = True
        
        # Start command processing thread
        self.processing_thread = threading.Thread(target=self._process_commands)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start listening thread
        self.listening_thread = threading.Thread(target=self._listen_for_commands)
        self.listening_thread.daemon = True
        self.listening_thread.start()
        
        self.speak("Voice assistant activated and ready for commands")
        logger.info("Voice assistant started")
        return True
    
    def stop(self):
        """Stop the voice assistant"""
        if not self.running:
            return False
        
        logger.info("Stopping voice assistant")
        self.running = False
        
        # Wait for threads to terminate
        if self.processing_thread:
            self.processing_thread.join(timeout=3.0)
        
        if self.listening_thread:
            self.listening_thread.join(timeout=3.0)
        
        self.speak("Voice assistant deactivated")
        logger.info("Voice assistant stopped")
        return True
    
    def speak(self, text):
        """Speak a message
        
        Args:
            text (str): Text to speak
        """
        logger.info(f"Speaking: {text}")
        
        # In a real implementation, this would use the TTS engine
        # self.engine.say(text)
        # self.engine.runAndWait()
        
        # For now, just print the text
        print(f"ASSISTANT: {text}")
    
    def process_command(self, command):
        """Process a voice command
        
        Args:
            command (str): Command text to process
        
        Returns:
            bool: True if the command was recognized and processed
        """
        if not command:
            return False
        
        # Convert to lowercase and clean up
        command = command.lower().strip()
        logger.info(f"Processing command: {command}")
        
        # Check against command patterns
        for pattern, handler in self.command_patterns.items():
            match = re.search(pattern, command)
            if match:
                logger.info(f"Command matched pattern: {pattern}")
                return handler(self, match)
        
        # No pattern matched
        self.speak("I'm sorry, I didn't understand that command.")
        return False
    
    def add_command_to_queue(self, command):
        """Add a command to the processing queue
        
        Args:
            command (str): Command text to add
        """
        self.command_queue.put(command)
    
    def _listen_for_commands(self):
        """Background thread for listening for voice commands"""
        logger.info("Listening thread started")
        
        try:
            while self.running:
                # In a real implementation, this would use the speech recognition library
                # with sr.Microphone() as source:
                #     self.recognizer.adjust_for_ambient_noise(source)
                #     print("Listening...")
                #     audio = self.recognizer.listen(source)
                #     try:
                #         text = self.recognizer.recognize_google(audio)
                #         if self.wake_phrase in text.lower():
                #             # Remove wake phrase from command
                #             command = text.lower().replace(self.wake_phrase, "").strip()
                #             if command:
                #                 self.add_command_to_queue(command)
                #             else:
                #                 self.speak("Yes, how can I help you?")
                #     except sr.UnknownValueError:
                #         pass
                #     except sr.RequestError as e:
                #         logger.error(f"Speech recognition service error: {e}")
                
                # For testing, simulate command input
                command = input("Enter a voice command (or 'exit' to stop): ")
                if command.lower() == 'exit':
                    self.running = False
                    break
                
                self.add_command_to_queue(command)
                
                # Sleep to avoid CPU overuse in the test mode
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in listening thread: {str(e)}")
        finally:
            logger.info("Listening thread stopped")
    
    def _process_commands(self):
        """Background thread for processing commands from the queue"""
        logger.info("Command processing thread started")
        
        try:
            while self.running:
                try:
                    # Get a command from the queue with a timeout
                    command = self.command_queue.get(timeout=1.0)
                    self.process_command(command)
                    self.command_queue.task_done()
                except queue.Empty:
                    # No commands in the queue, just continue
                    pass
                except Exception as e:
                    logger.error(f"Error processing command: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error in command processing thread: {str(e)}")
        finally:
            logger.info("Command processing thread stopped")
    
    def _initialize_command_patterns(self):
        """Initialize the command patterns and their handlers
        
        Returns:
            dict: Dictionary of regex patterns to handler functions
        """
        patterns = {
            # Basic interactions
            r'hello|hi|hey': self._cmd_greeting,
            r'goodbye|bye|exit|quit': self._cmd_goodbye,
            r'(what is|what\'s) your name': self._cmd_name,
            r'help|commands|what can you do': self._cmd_help,
            
            # Robotic arm control
            r'(move|turn) (to the |)?(left|right|up|down)': self._cmd_move_direction,
            r'(wave|greet|say hello)': self._cmd_wave,
            r'(nod|agree|say yes)': self._cmd_nod,
            r'point( at something|)': self._cmd_point,
            r'(grab|pick|pick up)( something|)': self._cmd_grab,
            r'(release|drop|let go)( something|)': self._cmd_release,
            r'(go|move) (to|to the|) home( position|)': self._cmd_home,
            
            # Environment scanning
            r'scan (the |)(area|environment|surroundings)': self._cmd_scan_area,
            r'(find|look for|search for) (nearest |closest |)(obstacle|object)': self._cmd_find_obstacle,
            r'(is|check if) (the |)(path|way) (is |)(clear|open|free)': self._cmd_check_path,
            
            # Object recognition
            r'(what|what\'s) (is |are |)(that|this|these|those)': self._cmd_identify_objects,
            r'(how many|count) (objects|things|items) (do you see|can you see|are there)': self._cmd_count_objects,
            r'(find|look for|search for) (a |the |)(person|people|human)': self._cmd_find_person,
            
            # Distance measurement
            r'(how|what is the|what\'s the) (far|distance) (to|between|from)': self._cmd_measure_distance,
        }
        
        return patterns
    
    # Command handlers
    def _cmd_greeting(self, match):
        """Handle greeting commands"""
        responses = [
            "Hello there!",
            "Hi, how can I assist you today?",
            "Greetings! How may I help you?"
        ]
        self.speak(random.choice(responses))
        return True
    
    def _cmd_goodbye(self, match):
        """Handle goodbye commands"""
        responses = [
            "Goodbye!",
            "See you later!",
            "Until next time!"
        ]
        self.speak(random.choice(responses))
        # Don't stop the assistant automatically, as this is just conversational
        return True
    
    def _cmd_name(self, match):
        """Handle name question"""
        self.speak("I am your robotic assistant, similar to Jarvis. You can call me RoboAssistant.")
        return True
    
    def _cmd_help(self, match):
        """Handle help command"""
        self.speak("I can help you control the robotic arm, scan the environment, identify objects, and measure distances. Try commands like 'move left', 'wave', 'scan the area', 'what is that', or 'how far to the nearest object'.")
        return True
    
    def _cmd_move_direction(self, match):
        """Handle directional movement commands"""
        if not self.robot_arm:
            self.speak("I cannot move the arm because it's not connected.")
            return False
        
        direction = match.group(3)
        
        self.speak(f"Moving the arm {direction}.")
        
        if direction == "left":
            current = self.robot_arm.joints["base"]
            self.robot_arm.move_joint("base", current + 20)
        elif direction == "right":
            current = self.robot_arm.joints["base"]
            self.robot_arm.move_joint("base", current - 20)
        elif direction == "up":
            current = self.robot_arm.joints["shoulder"]
            self.robot_arm.move_joint("shoulder", current - 20)
        elif direction == "down":
            current = self.robot_arm.joints["shoulder"]
            self.robot_arm.move_joint("shoulder", current + 20)
        
        return True
    
    def _cmd_wave(self, match):
        """Handle wave command"""
        if not self.robot_arm:
            self.speak("I cannot wave because the arm is not connected.")
            return False
        
        self.speak("Waving hello!")
        self.robot_arm.perform_gesture("wave")
        return True
    
    def _cmd_nod(self, match):
        """Handle nod command"""
        if not self.robot_arm:
            self.speak("I cannot nod because the arm is not connected.")
            return False
        
        self.speak("Nodding in agreement.")
        self.robot_arm.perform_gesture("nod")
        return True
    
    def _cmd_point(self, match):
        """Handle point command"""
        if not self.robot_arm:
            self.speak("I cannot point because the arm is not connected.")
            return False
        
        self.speak("Pointing ahead.")
        self.robot_arm.perform_gesture("point")
        return True
    
    def _cmd_grab(self, match):
        """Handle grab command"""
        if not self.robot_arm:
            self.speak("I cannot grab because the arm is not connected.")
            return False
        
        self.speak("Grabbing the object.")
        self.robot_arm.grab_object()
        return True
    
    def _cmd_release(self, match):
        """Handle release command"""
        if not self.robot_arm:
            self.speak("I cannot release because the arm is not connected.")
            return False
        
        self.speak("Releasing the object.")
        self.robot_arm.release_object()
        return True
    
    def _cmd_home(self, match):
        """Handle home position command"""
        if not self.robot_arm:
            self.speak("I cannot move to home position because the arm is not connected.")
            return False
        
        self.speak("Moving to home position.")
        self.robot_arm.move_to_home()
        return True
    
    def _cmd_scan_area(self, match):
        """Handle area scanning command"""
        self.speak("Scanning the area...")
        
        results = []
        
        # Use LIDAR to scan for obstacles
        if self.lidar:
            self.lidar.start_scanning()
            time.sleep(2)  # Allow time for scanning
            
            obstacles = self.lidar.detect_obstacles()
            if obstacles:
                results.append(f"I detected {len(obstacles)} obstacles.")
                
                nearest = self.lidar.get_nearest_obstacle()
                if nearest:
                    angle, distance = nearest
                    results.append(f"The nearest obstacle is {distance:.2f} meters away at {angle} degrees.")
            else:
                results.append("No obstacles detected with the LIDAR.")
                
            self.lidar.stop_scanning()
        
        # Use depth camera to detect objects
        if self.depth_camera:
            self.depth_camera.start_streaming()
            time.sleep(2)  # Allow time for frames
            
            objects = self.depth_camera.detect_objects()
            if objects:
                object_names = [obj.get("label", "unknown") for obj in objects]
                results.append(f"I can see {len(objects)} objects: {', '.join(object_names)}.")
            else:
                results.append("No objects detected with the depth camera.")
                
            self.depth_camera.stop_streaming()
        
        if not results:
            self.speak("I couldn't scan the area because no sensors are connected.")
            return False
        
        # Speak the combined results
        for result in results:
            self.speak(result)
            time.sleep(0.5)
        
        return True
    
    def _cmd_find_obstacle(self, match):
        """Handle find obstacle command"""
        if not self.lidar:
            self.speak("I cannot find obstacles because the LIDAR is not connected.")
            return False
        
        self.speak("Looking for the nearest obstacle...")
        
        self.lidar.start_scanning()
        time.sleep(1)  # Allow time for scanning
        
        nearest = self.lidar.get_nearest_obstacle()
        if nearest:
            angle, distance = nearest
            # Convert angle to direction
            direction = self._angle_to_direction(angle)
            self.speak(f"The nearest obstacle is {distance:.2f} meters to the {direction}.")
            
            # Point to the obstacle
            if self.robot_arm:
                self.speak("I'll point to it.")
                # Calculate arm position based on angle
                base_angle = self._lidar_angle_to_arm_angle(angle)
                self.robot_arm.move_joint("base", base_angle)
                self.robot_arm.perform_gesture("point")
        else:
            self.speak("I couldn't find any obstacles nearby.")
        
        self.lidar.stop_scanning()
        return True
    
    def _cmd_check_path(self, match):
        """Handle check path command"""
        if not self.lidar:
            self.speak("I cannot check the path because the LIDAR is not connected.")
            return False
        
        self.speak("Checking if the path ahead is clear...")
        
        self.lidar.start_scanning()
        time.sleep(1)  # Allow time for scanning
        
        # Check if the path ahead (0 degrees) is clear for 2 meters
        is_clear = self.lidar.is_path_clear(0, 2.0, 30)
        
        if is_clear:
            self.speak("The path ahead is clear for at least 2 meters.")
        else:
            self.speak("The path ahead is blocked within 2 meters.")
            
            # Try to find a clear path
            for angle in [30, -30, 60, -60, 90, -90]:
                if self.lidar.is_path_clear(angle, 2.0, 20):
                    direction = self._angle_to_direction(angle)
                    self.speak(f"However, there is a clear path to the {direction}.")
                    break
        
        self.lidar.stop_scanning()
        return True
    
    def _cmd_identify_objects(self, match):
        """Handle object identification command"""
        if not self.depth_camera:
            self.speak("I cannot identify objects because the depth camera is not connected.")
            return False
        
        self.speak("Looking at what's there...")
        
        self.depth_camera.start_streaming()
        time.sleep(1)  # Allow time for frames
        
        # In a real implementation, this would use the neural engine for detection
        # For now, simulate object detection
        objects = self.depth_camera.detect_objects()
        
        if objects:
            object_names = [obj.get("label", "unknown") for obj in objects]
            self.speak(f"I can see {len(objects)} objects: {', '.join(object_names)}.")
            
            # Describe the positions
            for i, obj in enumerate(objects):
                if i < 3:  # Limit to first 3 objects to avoid too much speech
                    label = obj.get("label", "unknown")
                    pos = obj.get("position", (0, 0, 0))
                    self.speak(f"The {label} is about {pos[2]:.2f} meters away.")
        else:
            self.speak("I don't see any recognizable objects right now.")
        
        self.depth_camera.stop_streaming()
        return True
    
    def _cmd_count_objects(self, match):
        """Handle count objects command"""
        if not self.depth_camera:
            self.speak("I cannot count objects because the depth camera is not connected.")
            return False
        
        self.speak("Counting objects...")
        
        self.depth_camera.start_streaming()
        time.sleep(1)  # Allow time for frames
        
        # In a real implementation, this would use the neural engine for detection
        # For now, simulate object detection
        objects = self.depth_camera.detect_objects()
        
        if objects:
            # Count objects by type
            counts = {}
            for obj in objects:
                label = obj.get("label", "unknown")
                counts[label] = counts.get(label, 0) + 1
            
            # Format the response
            total = len(objects)
            if total == 1:
                self.speak("I can see 1 object.")
            else:
                self.speak(f"I can see {total} objects.")
            
            # Describe the breakdown
            details = []
            for label, count in counts.items():
                if count == 1:
                    details.append(f"1 {label}")
                else:
                    details.append(f"{count} {label}s")
            
            if details:
                self.speak(f"That includes {', '.join(details)}.")
        else:
            self.speak("I don't see any objects right now.")
        
        self.depth_camera.stop_streaming()
        return True
    
    def _cmd_find_person(self, match):
        """Handle find person command"""
        if not self.depth_camera:
            self.speak("I cannot find people because the depth camera is not connected.")
            return False
        
        self.speak("Looking for people...")
        
        self.depth_camera.start_streaming()
        time.sleep(1)  # Allow time for frames
        
        # In a real implementation, this would use the neural engine for detection
        # For now, simulate object detection with a person
        objects = self.depth_camera.detect_objects()
        
        people = [obj for obj in objects if obj.get("label", "").lower() == "person"]
        
        if people:
            self.speak(f"I found {len(people)} {'person' if len(people) == 1 else 'people'}.")
            
            # Describe the positions of the first few people
            for i, person in enumerate(people):
                if i < 3:  # Limit to first 3 people
                    pos = person.get("position", (0, 0, 0))
                    direction = self._coords_to_direction(pos[0], pos[1])
                    self.speak(f"{'A person' if i == 0 else 'Another person'} is about {pos[2]:.2f} meters to the {direction}.")
            
            # Point to the nearest person
            if self.robot_arm and people:
                nearest = min(people, key=lambda p: p.get("position", (0, 0, 100))[2])
                pos = nearest.get("position", (0, 0, 0))
                angle = self._coords_to_angle(pos[0], pos[1])
                
                self.speak("I'll point to the nearest person.")
                base_angle = self._lidar_angle_to_arm_angle(angle)
                self.robot_arm.move_joint("base", base_angle)
                self.robot_arm.perform_gesture("point")
        else:
            self.speak("I don't see any people right now.")
        
        self.depth_camera.stop_streaming()
        return True
    
    def _cmd_measure_distance(self, match):
        """Handle distance measurement command"""
        if not self.depth_camera and not self.lidar:
            self.speak("I cannot measure distances because no sensors are connected.")
            return False
        
        self.speak("Measuring distance...")
        
        # For now, simulate a distance measurement
        # In a real implementation, this would use the depth camera or LIDAR
        
        if self.depth_camera:
            self.depth_camera.start_streaming()
            time.sleep(1)
            
            # Simulate getting depth at the center of the image
            depth = self.depth_camera.get_depth_at_point(320, 240)
            
            if depth:
                self.speak(f"The object directly in front is about {depth:.2f} meters away.")
            else:
                self.speak("I couldn't measure the distance to the object in front.")
            
            self.depth_camera.stop_streaming()
        
        elif self.lidar:
            self.lidar.start_scanning()
            time.sleep(1)
            
            # Get the nearest obstacle in the forward direction (+-15 degrees)
            scan = self.lidar.get_latest_scan()
            if scan:
                # Filter for points in the forward direction
                forward_points = [(angle, distance, quality) for angle, distance, quality in scan 
                                  if -15 <= angle <= 15 and quality > 10]
                
                if forward_points:
                    # Find the closest point
                    closest = min(forward_points, key=lambda p: p[1])
                    angle, distance, _ = closest
                    
                    self.speak(f"The nearest object directly ahead is about {distance:.2f} meters away.")
                else:
                    self.speak("I couldn't detect any objects directly ahead.")
            else:
                self.speak("I couldn't get a LIDAR scan.")
            
            self.lidar.stop_scanning()
        
        return True
    
    # Utility methods
    def _angle_to_direction(self, angle):
        """Convert an angle to a human-readable direction
        
        Args:
            angle (float): Angle in degrees (0-359)
        
        Returns:
            str: Direction as text (e.g., "front", "right", etc.)
        """
        if 337.5 <= angle or angle < 22.5:
            return "front"
        elif 22.5 <= angle < 67.5:
            return "front-right"
        elif 67.5 <= angle < 112.5:
            return "right"
        elif 112.5 <= angle < 157.5:
            return "back-right"
        elif 157.5 <= angle < 202.5:
            return "back"
        elif 202.5 <= angle < 247.5:
            return "back-left"
        elif 247.5 <= angle < 292.5:
            return "left"
        elif 292.5 <= angle < 337.5:
            return "front-left"
    
    def _coords_to_direction(self, x, y):
        """Convert coordinates to a human-readable direction
        
        Args:
            x (float): X coordinate (positive = right)
            y (float): Y coordinate (positive = forward)
        
        Returns:
            str: Direction as text
        """
        angle = math.degrees(math.atan2(y, x)) % 360
        return self._angle_to_direction(angle)
    
    def _coords_to_angle(self, x, y):
        """Convert coordinates to an angle
        
        Args:
            x (float): X coordinate (positive = right)
            y (float): Y coordinate (positive = forward)
        
        Returns:
            float: Angle in degrees (0-359)
        """
        return math.degrees(math.atan2(y, x)) % 360
    
    def _lidar_angle_to_arm_angle(self, lidar_angle):
        """Convert a LIDAR angle to a robotic arm base angle
        
        Args:
            lidar_angle (float): LIDAR angle in degrees (0-359)
        
        Returns:
            float: Arm base angle (0-180)
        """
        # LIDAR: 0 = forward, 90 = left, 180 = back, 270 = right
        # Arm: 0 = left, 90 = forward, 180 = right
        
        # Adjust the angle mapping based on your specific setup
        arm_angle = (90 - lidar_angle) % 180
        return arm_angle

# Test function
def test_voice_assistant():
    """Simple test function for the voice assistant"""
    from robotic_arm import RoboticArm
    from depth_camera import DepthCamera
    from lidar_scanner import LidarScanner
    
    # Initialize hardware interfaces
    arm = RoboticArm()
    camera = DepthCamera()
    lidar = LidarScanner()
    
    # Connect simulated hardware
    arm.connect()
    camera.connect()
    lidar.connect()
    
    # Create voice assistant
    assistant = VoiceAssistant(arm, camera, lidar)
    
    if assistant.setup():
        print("Voice assistant setup completed")
        
        if assistant.start():
            print("Voice assistant started")
            print("Try entering some commands (e.g. 'help', 'wave', 'scan the area')")
            print("Enter 'exit' to stop the test")
            
            # The assistant will run in the background threads
            # and process commands from the input() in the listening thread
            
            # Let it run for a while
            try:
                # In a real implementation, we wouldn't need this loop 
                # as the assistant runs in background threads
                while assistant.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("Test interrupted")
            finally:
                assistant.stop()
        
        print("Test completed")
    else:
        print("Failed to set up voice assistant")
    
    # Disconnect hardware
    arm.disconnect()
    camera.disconnect()
    lidar.disconnect()

if __name__ == "__main__":
    # Set up logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_voice_assistant() 