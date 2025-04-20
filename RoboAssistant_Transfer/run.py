#!/usr/bin/env python3
# RoboAssistant - Launcher Script
# Makes it easy to start the application with the correct settings

import os
import sys
import time
import json
import subprocess
import platform
import argparse
import logging
from pathlib import Path

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RoboAssistant.Launcher")

def load_config():
    """Load the configuration from settings.json"""
    try:
        config_path = Path('config') / 'settings.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        return None

def ensure_directories():
    """Ensure that required directories exist"""
    directories = ['logs', 'config', 'src', 'docs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

def check_hardware():
    """Check for connected hardware and update the configuration"""
    config = load_config()
    if not config:
        return False
    
    # For now, just print hardware status from configuration
    logger.info("Hardware status:")
    
    if config['hardware']['robotic_arm']['enabled']:
        logger.info(f"  - Robotic Arm: Enabled (Port: {config['hardware']['robotic_arm']['port']})")
    else:
        logger.info("  - Robotic Arm: Disabled")
    
    if config['hardware']['depth_camera']['enabled']:
        logger.info("  - Depth Camera: Enabled")
    else:
        logger.info("  - Depth Camera: Disabled")
    
    if config['hardware']['lidar']['enabled']:
        logger.info(f"  - LIDAR Scanner: Enabled (Port: {config['hardware']['lidar']['port']})")
    else:
        logger.info("  - LIDAR Scanner: Disabled")
    
    if config['hardware']['coral_accelerator']['enabled']:
        logger.info("  - Coral USB Accelerator: Enabled")
    else:
        logger.info("  - Coral USB Accelerator: Disabled")
    
    return True

def check_dependencies():
    """Check if Python dependencies are installed"""
    try:
        # This could be more sophisticated by actually importing the modules
        # or checking pip list, but for now, just check if requirements.txt exists
        if os.path.exists('requirements.txt'):
            logger.info("Dependencies file found: requirements.txt")
            
            # Ask if user wants to install dependencies
            install = input("Would you like to install/update dependencies? (y/n): ")
            if install.lower() == 'y':
                logger.info("Installing dependencies...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                logger.info("Dependencies installed successfully")
        else:
            logger.warning("requirements.txt not found. Dependencies might be missing.")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Failed to check/install dependencies: {str(e)}")
        return False

def start_application(mode=None):
    """Start the main application"""
    try:
        # Build the command to run the application
        main_script = os.path.join('src', 'main.py')
        if not os.path.exists(main_script):
            logger.error(f"Main script not found: {main_script}")
            return False
        
        command = [sys.executable, main_script]
        
        # Add the mode argument if specified
        if mode:
            command.extend(["--mode", mode])
        
        # Print the command for the user
        cmd_str = ' '.join(command)
        logger.info(f"Starting application with command: {cmd_str}")
        
        # Start the application
        process = subprocess.Popen(command)
        
        # Wait for a moment to ensure it starts
        time.sleep(1)
        if process.poll() is None:
            logger.info("Application started successfully")
            return True
        else:
            logger.error("Application failed to start")
            return False
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        return False

def main():
    """Main function for the launcher"""
    parser = argparse.ArgumentParser(description='RoboAssistant Launcher')
    parser.add_argument('--mode', choices=['assistant', 'scanner', 'surveillance'],
                        help='Start the application in a specific mode')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check hardware and dependencies, don\'t start the application')
    args = parser.parse_args()
    
    # Display banner
    print("\n" + "="*60)
    print(" RoboAssistant - Intelligent Robotic Control System")
    print("="*60)
    print(f" System: {platform.system()} {platform.release()}")
    print(f" Python: {platform.python_version()}")
    print("="*60 + "\n")
    
    # Ensure directories exist
    ensure_directories()
    
    # Check for hardware connections
    logger.info("Checking hardware connections...")
    if not check_hardware():
        logger.warning("Hardware check failed. Some features may not work correctly.")
    
    # Check dependencies
    logger.info("Checking dependencies...")
    if not check_dependencies():
        logger.warning("Dependency check failed. Some features may not work correctly.")
    
    # If check-only flag is set, don't start the application
    if args.check_only:
        logger.info("Check-only mode is set. Not starting application.")
        return
    
    # Start the application
    logger.info("Starting RoboAssistant...")
    if not start_application(args.mode):
        logger.error("Failed to start RoboAssistant.")
        return
    
    logger.info("Launcher completed successfully.")

if __name__ == "__main__":
    main() 