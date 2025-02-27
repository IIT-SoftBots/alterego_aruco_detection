# ArUco Marker Detection for AlterEGO Robot

## Overview
This package provides ArUco marker detection functionality using a ZED camera for the AlterEGO robot. It detects ArUco markers and publishes their 6D pose relative to the camera.

## Dependencies
- ROS Noetic

## Installation
```bash
# Clone the repository
cd ~/catkin_ws/src
git clone <repo_url> alterego_aruco_detection

# Install dependencies
sudo apt-get install ros-noetic-cv-bridge python3-opencv

# Build
cd ~/catkin_ws
catkin_make