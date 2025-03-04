cmake_minimum_required(VERSION 3.0.2)
project(alterego_aruco_detection)

## Find catkin and all required components
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  std_msgs
  sensor_msgs
  image_transport
  message_generation
  alterego_msgs
)

## Find OpenCV
find_package(OpenCV REQUIRED)

## Declare catkin package
catkin_package(
  CATKIN_DEPENDS
    roscpp
    rospy
    cv_bridge
    std_msgs
    sensor_msgs
    alterego_msgs
)

###########
## Build ##
###########

## Specify include directories
include_directories(
  # include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Mark executable scripts for installation
catkin_install_python(PROGRAMS
  src/aruco_detector.py
  src/publish_marker_goal.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)