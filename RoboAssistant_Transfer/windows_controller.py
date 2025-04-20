#!/usr/bin/env python3
# RoboAssistant - Windows Controller Application
# Remote control interface for the Raspberry Pi robot

import sys
import os
import json
import time
import socket
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import PIL.Image, PIL.ImageTk
import cv2
import numpy as np
from io import BytesIO
import base64
import requests

class RobotControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RoboAssistant Control Panel")
        self.root.minsize(800, 600)
        
        # Config
        self.raspberry_pi_ip = "192.168.1.100"  # Default IP, will be configurable
        self.raspberry_pi_port = 5000  # Default port
        self.connected = False
        self.current_mode = "standby"
        
        # Communication queue
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Create the UI
        self.create_ui()
        
        # Start the communication thread
        self.comm_thread = threading.Thread(target=self.communication_thread, daemon=True)
        self.comm_thread.start()
        
        # Regularly update the UI
        self.update_ui()
    
    def create_ui(self):
        """Create the user interface"""
        # Main frame with tabs
        self.tab_control = ttk.Notebook(self.root)
        
        # Connection tab
        self.connection_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.connection_tab, text="Connection")
        
        # Control tab
        self.control_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.control_tab, text="Robot Control")
        
        # Camera tab
        self.camera_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.camera_tab, text="Camera View")
        
        # LIDAR tab
        self.lidar_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.lidar_tab, text="LIDAR View")
        
        # Settings tab
        self.settings_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.settings_tab, text="Settings")
        
        self.tab_control.pack(expand=1, fill="both")
        
        # Connection tab content
        self.setup_connection_tab()
        
        # Control tab content
        self.setup_control_tab()
        
        # Camera tab content
        self.setup_camera_tab()
        
        # LIDAR tab content
        self.setup_lidar_tab()
        
        # Settings tab content
        self.setup_settings_tab()
        
        # Status bar at the bottom
        self.status_bar = ttk.Label(self.root, text="Status: Disconnected", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_connection_tab(self):
        """Set up the connection tab"""
        frame = ttk.Frame(self.connection_tab, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # IP Address
        ttk.Label(frame, text="Raspberry Pi IP:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.ip_entry = ttk.Entry(frame, width=15)
        self.ip_entry.insert(0, self.raspberry_pi_ip)
        self.ip_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Port
        ttk.Label(frame, text="Port:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.port_entry = ttk.Entry(frame, width=5)
        self.port_entry.insert(0, str(self.raspberry_pi_port))
        self.port_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Connect/Disconnect button
        self.connect_button = ttk.Button(frame, text="Connect", command=self.toggle_connection)
        self.connect_button.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Connection status
        self.conn_status_lbl = ttk.Label(frame, text="Disconnected", foreground="red")
        self.conn_status_lbl.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Log window
        ttk.Label(frame, text="Connection Log:").grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)
        self.log_text = scrolledtext.ScrolledText(frame, height=10)
        self.log_text.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.log_text.configure(state='disabled')
        
        # Info about client-server mode
        info_text = """
        This application connects to the RoboAssistant software running on your Raspberry Pi.
        
        To use this controller:
        1. Make sure the RoboAssistant is running on your Raspberry Pi in server mode
        2. Enter the IP address of your Raspberry Pi
        3. Click the Connect button
        
        Server mode can be started on the Raspberry Pi with:
        python run.py --server
        """
        info_label = ttk.Label(frame, text=info_text, justify=tk.LEFT, wraplength=400)
        info_label.grid(row=6, column=0, columnspan=2, pady=10, sticky=tk.W)
    
    def setup_control_tab(self):
        """Set up the robot control tab"""
        frame = ttk.Frame(self.control_tab, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Mode selection
        ttk.Label(frame, text="Operating Mode:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.mode_var = tk.StringVar(value="assistant")
        mode_combo = ttk.Combobox(frame, textvariable=self.mode_var, 
                                  values=["assistant", "scanner", "surveillance"])
        mode_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        ttk.Button(frame, text="Switch Mode", command=self.switch_mode).grid(row=0, column=2, padx=5)
        
        # Arm controls
        arm_frame = ttk.LabelFrame(frame, text="Robotic Arm Control", padding=10)
        arm_frame.grid(row=1, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Arm movement controls
        ttk.Button(arm_frame, text="↑", width=3, command=lambda: self.move_arm("up")).grid(row=0, column=1)
        ttk.Button(arm_frame, text="←", width=3, command=lambda: self.move_arm("left")).grid(row=1, column=0)
        ttk.Button(arm_frame, text="→", width=3, command=lambda: self.move_arm("right")).grid(row=1, column=2)
        ttk.Button(arm_frame, text="↓", width=3, command=lambda: self.move_arm("down")).grid(row=2, column=1)
        
        # Home position
        ttk.Button(arm_frame, text="Home", command=lambda: self.move_arm("home")).grid(row=1, column=1)
        
        # Gripper controls
        ttk.Button(arm_frame, text="Grab", command=self.grab).grid(row=0, column=3, padx=10)
        ttk.Button(arm_frame, text="Release", command=self.release).grid(row=1, column=3, padx=10)
        
        # Gesture controls
        ttk.Button(arm_frame, text="Wave", command=lambda: self.perform_gesture("wave")).grid(row=0, column=4, padx=10)
        ttk.Button(arm_frame, text="Nod", command=lambda: self.perform_gesture("nod")).grid(row=1, column=4, padx=10)
        ttk.Button(arm_frame, text="Point", command=lambda: self.perform_gesture("point")).grid(row=2, column=4, padx=10)
        
        # Voice command input
        voice_frame = ttk.LabelFrame(frame, text="Voice Commands", padding=10)
        voice_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        ttk.Label(voice_frame, text="Enter command:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.command_entry = ttk.Entry(voice_frame, width=40)
        self.command_entry.grid(row=0, column=1, padx=5)
        ttk.Button(voice_frame, text="Send", command=self.send_voice_command).grid(row=0, column=2, padx=5)
        
        # Command output
        ttk.Label(frame, text="System Response:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.response_text = scrolledtext.ScrolledText(frame, height=8)
        self.response_text.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))
        self.response_text.configure(state='disabled')
    
    def setup_camera_tab(self):
        """Set up the camera view tab"""
        frame = ttk.Frame(self.camera_tab, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Camera controls
        control_frame = ttk.Frame(frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Start Camera", command=self.start_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop Camera", command=self.stop_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Take Snapshot", command=self.take_snapshot).pack(side=tk.LEFT, padx=5)
        
        self.depth_view_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Show Depth View", variable=self.depth_view_var, 
                        command=self.toggle_depth_view).pack(side=tk.LEFT, padx=15)
        
        # Camera view
        self.camera_view_label = ttk.Label(frame, text="Camera view will appear here")
        self.camera_view_label.pack(expand=True, fill=tk.BOTH, pady=10)
        
        # Object detection
        detect_frame = ttk.Frame(frame)
        detect_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        ttk.Button(detect_frame, text="Detect Objects", command=self.detect_objects).pack(side=tk.LEFT, padx=5)
        self.detection_result_label = ttk.Label(detect_frame, text="No objects detected")
        self.detection_result_label.pack(side=tk.LEFT, padx=10)
    
    def setup_lidar_tab(self):
        """Set up the LIDAR view tab"""
        frame = ttk.Frame(self.lidar_tab, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # LIDAR controls
        control_frame = ttk.Frame(frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Start LIDAR", command=self.start_lidar).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop LIDAR", command=self.stop_lidar).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Create Map", command=self.create_map).pack(side=tk.LEFT, padx=5)
        
        # LIDAR view (will show a polar plot of LIDAR data)
        self.lidar_view_label = ttk.Label(frame, text="LIDAR data will appear here")
        self.lidar_view_label.pack(expand=True, fill=tk.BOTH, pady=10)
        
        # Obstacle detection
        obstacle_frame = ttk.Frame(frame)
        obstacle_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        ttk.Button(obstacle_frame, text="Detect Obstacles", command=self.detect_obstacles).pack(side=tk.LEFT, padx=5)
        ttk.Button(obstacle_frame, text="Check Path", command=self.check_path).pack(side=tk.LEFT, padx=5)
        self.obstacle_result_label = ttk.Label(obstacle_frame, text="No obstacles detected")
        self.obstacle_result_label.pack(side=tk.LEFT, padx=10)
    
    def setup_settings_tab(self):
        """Set up the settings tab"""
        frame = ttk.Frame(self.settings_tab, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # About section
        about_frame = ttk.LabelFrame(frame, text="About", padding=10)
        about_frame.pack(fill=tk.X, pady=10)
        
        about_text = """
        RoboAssistant Windows Controller
        
        This application provides a user-friendly interface to control 
        the RoboAssistant robotic system running on a Raspberry Pi.
        
        Features:
        - Remote control of the robotic arm
        - Voice command input
        - Live camera feed viewing
        - LIDAR data visualization
        - Multiple operation modes
        """
        about_label = ttk.Label(about_frame, text=about_text, justify=tk.LEFT, wraplength=500)
        about_label.pack(pady=5)
        
        # Settings
        settings_frame = ttk.LabelFrame(frame, text="Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=10)
        
        # Auto-connect on startup
        self.autoconnect_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Auto-connect on startup", 
                        variable=self.autoconnect_var).pack(anchor=tk.W, pady=5)
        
        # Auto-reconnect if connection lost
        self.autoreconnect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Auto-reconnect if connection lost", 
                        variable=self.autoreconnect_var).pack(anchor=tk.W, pady=5)
        
        # Save settings button
        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings).pack(pady=10)
        
        # Server script
        server_frame = ttk.LabelFrame(frame, text="Raspberry Pi Server Setup", padding=10)
        server_frame.pack(fill=tk.X, pady=10)
        
        server_text = """
        To enable remote control, you need to run the server script on your Raspberry Pi.
        
        Copy these files to your Raspberry Pi:
        - remote_server.py - The main server script
        - src/ - The hardware interface modules
        
        Then run:
        python3 remote_server.py
        
        Make sure all hardware is connected to the Raspberry Pi before starting the server.
        """
        server_label = ttk.Label(server_frame, text=server_text, justify=tk.LEFT, wraplength=500)
        server_label.pack(pady=5)
    
    # UI update method
    def update_ui(self):
        """Update the UI elements periodically"""
        # Process any messages in the response queue
        try:
            while not self.response_queue.empty():
                response = self.response_queue.get_nowait()
                self.process_response(response)
        except queue.Empty:
            pass
        
        # Schedule the next update
        self.root.after(100, self.update_ui)
    
    # Connection methods
    def toggle_connection(self):
        """Toggle the connection to the Raspberry Pi"""
        if not self.connected:
            # Update IP and port from UI
            self.raspberry_pi_ip = self.ip_entry.get()
            self.raspberry_pi_port = int(self.port_entry.get())
            
            # Attempt to connect
            self.log_message(f"Connecting to {self.raspberry_pi_ip}:{self.raspberry_pi_port}...")
            
            try:
                # In a real implementation, this would establish a connection
                # with the server running on the Raspberry Pi
                # For now, we'll simulate a successful connection
                
                # self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # self.client_socket.connect((self.raspberry_pi_ip, self.raspberry_pi_port))
                
                self.connected = True
                self.connect_button.config(text="Disconnect")
                self.conn_status_lbl.config(text="Connected", foreground="green")
                self.status_bar.config(text=f"Status: Connected to {self.raspberry_pi_ip}")
                self.log_message("Connection established")
                
                # Send initial status request
                self.send_command("status")
            except Exception as e:
                self.log_message(f"Connection failed: {str(e)}")
                messagebox.showerror("Connection Error", f"Failed to connect: {str(e)}")
        else:
            # Disconnect
            self.log_message("Disconnecting...")
            
            try:
                # In a real implementation, this would close the connection
                # self.client_socket.close()
                
                self.connected = False
                self.connect_button.config(text="Connect")
                self.conn_status_lbl.config(text="Disconnected", foreground="red")
                self.status_bar.config(text="Status: Disconnected")
                self.log_message("Disconnected")
            except Exception as e:
                self.log_message(f"Error during disconnect: {str(e)}")
    
    def log_message(self, message):
        """Add a message to the log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')
    
    # Communication thread
    def communication_thread(self):
        """Background thread for communication with the Raspberry Pi"""
        while True:
            # Check if we have commands to send
            try:
                if self.connected and not self.command_queue.empty():
                    command = self.command_queue.get()
                    
                    # In a real implementation, this would send the command to the server
                    # and receive a response
                    # self.client_socket.send(json.dumps(command).encode())
                    # response = self.client_socket.recv(4096).decode()
                    
                    # For now, simulate responses
                    response = self.simulate_response(command)
                    
                    # Add response to queue for UI thread to process
                    self.response_queue.put(response)
            except Exception as e:
                self.log_message(f"Communication error: {str(e)}")
                if self.connected and self.autoreconnect_var.get():
                    self.log_message("Attempting to reconnect...")
                    # Attempt reconnection logic would go here
            
            # Sleep to avoid CPU overuse
            time.sleep(0.1)
    
    def send_command(self, command_type, **kwargs):
        """Send a command to the Raspberry Pi"""
        if not self.connected:
            self.log_message("Cannot send command: Not connected")
            return
        
        command = {
            "type": command_type,
            "timestamp": time.time(),
            **kwargs
        }
        
        self.command_queue.put(command)
    
    def simulate_response(self, command):
        """Simulate a response from the server (for testing)"""
        command_type = command.get("type", "")
        
        if command_type == "status":
            return {
                "type": "status",
                "status": "ok",
                "mode": "assistant",
                "devices": {
                    "robotic_arm": True,
                    "depth_camera": True,
                    "lidar": True,
                    "neural_engine": True
                }
            }
        
        elif command_type == "arm_move":
            direction = command.get("direction", "")
            return {
                "type": "arm_move",
                "status": "ok",
                "message": f"Arm moved {direction}"
            }
        
        elif command_type == "gesture":
            gesture = command.get("gesture", "")
            return {
                "type": "gesture",
                "status": "ok",
                "message": f"Performed gesture: {gesture}"
            }
        
        elif command_type == "gripper":
            action = command.get("action", "")
            return {
                "type": "gripper",
                "status": "ok",
                "message": f"Gripper {action}"
            }
        
        elif command_type == "voice_command":
            cmd = command.get("command", "")
            return {
                "type": "voice_command",
                "status": "ok",
                "message": f"Processed command: {cmd}",
                "response": f"I understood: {cmd}"
            }
        
        elif command_type == "switch_mode":
            mode = command.get("mode", "")
            return {
                "type": "switch_mode",
                "status": "ok",
                "mode": mode,
                "message": f"Switched to {mode} mode"
            }
        
        else:
            return {
                "type": "unknown",
                "status": "error",
                "message": f"Unknown command type: {command_type}"
            }
    
    def process_response(self, response):
        """Process a response from the server"""
        response_type = response.get("type", "")
        status = response.get("status", "")
        message = response.get("message", "")
        
        if status == "error":
            self.log_message(f"Error: {message}")
        
        if response_type == "status":
            # Update UI with device status
            if "mode" in response:
                self.current_mode = response["mode"]
                self.mode_var.set(self.current_mode)
        
        elif response_type == "voice_command":
            # Display the response in the text box
            if "response" in response:
                self.update_response_text(response["response"])
        
        elif response_type == "switch_mode":
            # Update the current mode
            if "mode" in response:
                self.current_mode = response["mode"]
                self.mode_var.set(self.current_mode)
                self.update_response_text(f"Switched to {self.current_mode} mode")
    
    def update_response_text(self, text):
        """Update the response text area"""
        self.response_text.configure(state='normal')
        self.response_text.insert(tk.END, f"{text}\n")
        self.response_text.see(tk.END)
        self.response_text.configure(state='disabled')
    
    # Control methods
    def switch_mode(self):
        """Switch the operating mode"""
        new_mode = self.mode_var.get()
        self.send_command("switch_mode", mode=new_mode)
    
    def move_arm(self, direction):
        """Move the robotic arm"""
        self.send_command("arm_move", direction=direction)
        self.update_response_text(f"Moving arm {direction}")
    
    def grab(self):
        """Close the gripper"""
        self.send_command("gripper", action="grab")
        self.update_response_text("Closing gripper")
    
    def release(self):
        """Open the gripper"""
        self.send_command("gripper", action="release")
        self.update_response_text("Opening gripper")
    
    def perform_gesture(self, gesture):
        """Perform a gesture"""
        self.send_command("gesture", gesture=gesture)
        self.update_response_text(f"Performing {gesture} gesture")
    
    def send_voice_command(self):
        """Send a voice command"""
        command = self.command_entry.get()
        if command:
            self.send_command("voice_command", command=command)
            self.update_response_text(f"You: {command}")
            self.command_entry.delete(0, tk.END)
    
    # Camera methods
    def start_camera(self):
        """Start the camera stream"""
        self.send_command("camera", action="start")
        self.update_response_text("Starting camera stream")
    
    def stop_camera(self):
        """Stop the camera stream"""
        self.send_command("camera", action="stop")
        self.update_response_text("Stopping camera stream")
    
    def take_snapshot(self):
        """Take a camera snapshot"""
        self.send_command("camera", action="snapshot")
        self.update_response_text("Taking snapshot")
    
    def toggle_depth_view(self):
        """Toggle between RGB and depth view"""
        show_depth = self.depth_view_var.get()
        self.send_command("camera", action="set_view", depth=show_depth)
        view_type = "depth" if show_depth else "RGB"
        self.update_response_text(f"Switching to {view_type} view")
    
    def detect_objects(self):
        """Detect objects in the camera view"""
        self.send_command("camera", action="detect")
        self.update_response_text("Detecting objects")
    
    # LIDAR methods
    def start_lidar(self):
        """Start the LIDAR scanner"""
        self.send_command("lidar", action="start")
        self.update_response_text("Starting LIDAR scanner")
    
    def stop_lidar(self):
        """Stop the LIDAR scanner"""
        self.send_command("lidar", action="stop")
        self.update_response_text("Stopping LIDAR scanner")
    
    def create_map(self):
        """Create a map from LIDAR data"""
        self.send_command("lidar", action="create_map")
        self.update_response_text("Creating map from LIDAR data")
    
    def detect_obstacles(self):
        """Detect obstacles with LIDAR"""
        self.send_command("lidar", action="detect_obstacles")
        self.update_response_text("Detecting obstacles")
    
    def check_path(self):
        """Check if path is clear"""
        self.send_command("lidar", action="check_path", angle=0, distance=2.0)
        self.update_response_text("Checking if path is clear")
    
    # Settings methods
    def save_settings(self):
        """Save the current settings"""
        settings = {
            "raspberry_pi_ip": self.raspberry_pi_ip,
            "raspberry_pi_port": self.raspberry_pi_port,
            "autoconnect": self.autoconnect_var.get(),
            "autoreconnect": self.autoreconnect_var.get()
        }
        
        try:
            with open("controller_settings.json", "w") as f:
                json.dump(settings, f, indent=4)
            messagebox.showinfo("Settings", "Settings saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")

def main():
    root = tk.Tk()
    app = RobotControlApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 