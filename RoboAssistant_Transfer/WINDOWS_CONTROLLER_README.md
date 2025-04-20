# RoboAssistant Windows Controller

A Windows application for remotely controlling your Raspberry Pi-based robotic system.

## Overview

This application allows you to control your RoboAssistant robotic system from a Windows PC over the network. It provides a user-friendly interface for:

- Controlling the robotic arm
- Sending voice commands
- Viewing camera feeds
- Monitoring LIDAR data
- Switching between operating modes

## Setup

### Prerequisites

On your Windows PC:
- Python 3.7 or higher
- The following Python packages:
  - `tkinter` (usually comes with Python)
  - `pillow` (PIL)
  - `numpy`
  - `opencv-python`
  - `requests`

You can install them with:
```
pip install pillow numpy opencv-python requests
```

### Raspberry Pi Setup

Before using the Windows controller, you need to set up your Raspberry Pi:

1. Copy all RoboAssistant files to your Raspberry Pi
2. Make sure all hardware components are connected:
   - Waveshare Robotic Arm
   - Intel RealSense D455 camera
   - RPLIDAR A1M8
   - Coral USB Accelerator
3. Install required dependencies on the Raspberry Pi (see main README.md)
4. Start the remote server on the Raspberry Pi:
   ```
   python3 remote_server.py
   ```

### Windows Controller Setup

1. Copy the following files to your Windows PC:
   - `windows_controller.py`
   - `controller_settings.json` (will be created on first run)

## Usage

### Starting the Controller

Run the Windows controller:
```
python windows_controller.py
```

### Connecting to the Raspberry Pi

1. Go to the "Connection" tab
2. Enter the IP address of your Raspberry Pi (e.g., 192.168.1.100)
3. The default port is 5000
4. Click "Connect"

### Controlling the Robot

#### Robot Control Tab

- Use the directional buttons to move the arm
- Use "Grab" and "Release" to control the gripper
- Use gesture buttons for predefined movements
- Enter voice commands in the text field

#### Camera View Tab

- Click "Start Camera" to begin the video feed
- Switch between RGB and depth views
- Take snapshots
- Run object detection

#### LIDAR View Tab

- Start and stop the LIDAR scanner
- Create maps from LIDAR data
- Detect obstacles
- Check if paths are clear

### Settings

The Settings tab allows you to configure:
- Auto-connect on startup
- Auto-reconnect if connection is lost

## Troubleshooting

1. **Connection Errors**
   - Make sure the Raspberry Pi and Windows PC are on the same network
   - Check that the remote_server.py is running on the Raspberry Pi
   - Verify there are no firewall restrictions blocking port 5000

2. **No Hardware Detection**
   - Check physical connections of hardware to the Raspberry Pi
   - Verify hardware is enabled in the config/settings.json file on the Pi

3. **Camera or LIDAR Not Working**
   - Make sure the proper drivers are installed on the Raspberry Pi
   - Check device permissions on the Raspberry Pi

## Development

This application uses a client-server architecture:
- The Windows controller is the client
- The remote_server.py on the Raspberry Pi is the server

Communication is done via TCP sockets with JSON messages.

## License

MIT License 