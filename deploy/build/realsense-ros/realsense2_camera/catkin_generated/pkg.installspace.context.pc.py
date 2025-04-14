# generated from catkin/cmake/template/pkg.context.pc.in
CATKIN_PACKAGE_PREFIX = ""
PROJECT_PKG_CONFIG_INCLUDE_DIRS = "${prefix}/include".split(';') if "${prefix}/include" != "" else []
PROJECT_CATKIN_DEPENDS = "message_runtime;roscpp;sensor_msgs;std_msgs;nodelet;cv_bridge;image_transport;ddynamic_reconfigure;nav_msgs".replace(';', ' ')
PKG_CONFIG_LIBRARIES_WITH_PREFIX = "-lrealsense2_camera".split(';') if "-lrealsense2_camera" != "" else []
PROJECT_NAME = "realsense2_camera"
PROJECT_SPACE_DIR = "/home/zxh/ws/wrqws/quaddif/deploy/install"
PROJECT_VERSION = "2.3.2"
