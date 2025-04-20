# RoboAssistant

A comprehensive software system for controlling the Waveshare Robotic Arm (RoArm M2-S) with additional sensors including Intel RealSense D455 camera, RPLIDAR A1M8, and Google Coral USB Accelerator, all managed by a Raspberry Pi 5.

## Features

- **Voice Control**: Jarvis-like assistant that listens to commands and controls the robotic arm
- **3D Scanning**: Creates detailed 3D meshes using both the Intel RealSense camera and LIDAR
- **Object Detection**: Identifies objects and avoids collisions using the depth camera and LIDAR
- **Area Surveillance**: Scans an area and detects if objects are not in their correct positions
- **Quality Control**: Combines RealSense and LIDAR data to create high-precision scans and compare them with Fusion 360 models for QC checks

## Hardware Requirements

- Waveshare Robotic Arm (RoArm M2-S)
- Intel RealSense D455 depth camera
- RPLIDAR A1M8 360° scanner
- Raspberry Pi 5
- Google Coral USB Accelerator

## Software Setup

### Installation

1. Clone this repository:
   ```
git clone https://github.com/Shanroy1/RoboAssistant.git
cd RoboAssistant
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Connect the hardware devices:
   - Connect the Robotic Arm to a USB port (typically COM3 or /dev/ttyUSB0)
   - Connect the Intel RealSense camera via USB 3.0
   - Connect the LIDAR scanner to another USB port (typically COM4 or /dev/ttyUSB1)
   - Connect the Coral USB Accelerator to a USB 3.0 port

4. Adjust the settings in `config/settings.json` to match your hardware configuration

### Running the Application

Start the main application:
```
python src/main.py
```

Or use the launcher script:
```
python run.py
```

Select from available modes:
1. Assistant Mode (Jarvis-like voice control)
2. Scanner Mode (3D Scanning)
3. Surveillance Mode (Area Monitoring)
4. QC Mode (Quality Control)

## Usage Examples

### Voice Commands

The assistant responds to various voice commands such as:
- "Wave hello" - Makes the arm wave
- "Grab that object" - Closes the gripper to grab an object
- "Scan the area" - Performs a 360° scan with the LIDAR
- "What is that?" - Identifies objects using the camera and neural processing

### 3D Scanning

In scanning mode, the system:
1. Uses the Intel RealSense camera to capture depth maps
2. Uses the LIDAR to scan the environment
3. Combines both data sources to create a comprehensive 3D model
4. Saves the scan as a standard 3D model format

### Surveillance

In surveillance mode, the system:
1. Creates a baseline scan of the area
2. Continuously monitors for changes
3. Alerts when objects are moved or removed

### Quality Control

In QC mode, the system:
1. Calibrates the RealSense camera and LIDAR to ensure alignment
2. Creates high-precision combined scans from both sensors
3. Imports reference models from Fusion 360 (STL, OBJ)
4. Aligns the scan with the model and measures deviations
5. Generates color-coded visualizations and detailed reports
6. Provides pass/fail status based on configurable tolerances

## Remote Control

### Windows Controller

A Windows GUI application is provided for remote control of the robotic system:
- Control the arm from your PC
- View camera feeds
- Monitor LIDAR data
- Run QC checks

To use the Windows controller:
1. Run the server on the Raspberry Pi: `python remote_server.py`
2. Run the Windows controller: `python windows_controller.py`

See the `WINDOWS_CONTROLLER_README.md` for detailed setup instructions.

## Project Structure

- `src/` - Main source code
  - `main.py` - Entry point
  - `robotic_arm.py` - Robotic arm interface
  - `depth_camera.py` - Intel RealSense interface
  - `lidar_scanner.py` - LIDAR interface
  - `neural_engine.py` - Coral USB Accelerator interface
  - `voice_assistant.py` - Voice assistant functionality
  - `fusion_qc.py` - Quality control functionality
- `config/` - Configuration files
- `logs/` - Application logs
- `docs/` - Documentation
- `output/` - Output files
  - `scans/` - 3D scan files
  - `reports/` - QC reports
  - `visualizations/` - QC visualizations

## License

MIT License

## Acknowledgments

- Waveshare for the RoArm M2-S
- Intel for the RealSense SDK
- Google for the Coral USB Accelerator
- The Raspberry Pi Foundation 
