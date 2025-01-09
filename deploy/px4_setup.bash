#! /bin/bash

source $PX4_PATH/Tools/simulation/gazebo-classic/setup_gazebo.bash $PX4_PATH $PX4_PATH/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$PX4_PATH
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$PX4_PATH/Tools/simulation/gazebo-classic/sitl_gazebo-classic
echo ROS_PACKAGE_PATH = $ROS_PACKAGE_PATH