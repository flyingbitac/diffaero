# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "realsense2_camera: 3 messages, 1 services")

set(MSG_I_FLAGS "-Irealsense2_camera:/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg;-Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg;-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(realsense2_camera_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/IMUInfo.msg" NAME_WE)
add_custom_target(_realsense2_camera_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "realsense2_camera" "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/IMUInfo.msg" ""
)

get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Extrinsics.msg" NAME_WE)
add_custom_target(_realsense2_camera_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "realsense2_camera" "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Extrinsics.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Metadata.msg" NAME_WE)
add_custom_target(_realsense2_camera_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "realsense2_camera" "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Metadata.msg" "std_msgs/Header"
)

get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/srv/DeviceInfo.srv" NAME_WE)
add_custom_target(_realsense2_camera_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "realsense2_camera" "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/srv/DeviceInfo.srv" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages
_generate_msg_cpp(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/IMUInfo.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/realsense2_camera
)
_generate_msg_cpp(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Extrinsics.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/realsense2_camera
)
_generate_msg_cpp(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Metadata.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/realsense2_camera
)

### Generating Services
_generate_srv_cpp(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/srv/DeviceInfo.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/realsense2_camera
)

### Generating Module File
_generate_module_cpp(realsense2_camera
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/realsense2_camera
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(realsense2_camera_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(realsense2_camera_generate_messages realsense2_camera_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/IMUInfo.msg" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_cpp _realsense2_camera_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Extrinsics.msg" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_cpp _realsense2_camera_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Metadata.msg" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_cpp _realsense2_camera_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/srv/DeviceInfo.srv" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_cpp _realsense2_camera_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(realsense2_camera_gencpp)
add_dependencies(realsense2_camera_gencpp realsense2_camera_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS realsense2_camera_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages
_generate_msg_eus(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/IMUInfo.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/realsense2_camera
)
_generate_msg_eus(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Extrinsics.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/realsense2_camera
)
_generate_msg_eus(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Metadata.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/realsense2_camera
)

### Generating Services
_generate_srv_eus(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/srv/DeviceInfo.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/realsense2_camera
)

### Generating Module File
_generate_module_eus(realsense2_camera
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/realsense2_camera
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(realsense2_camera_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(realsense2_camera_generate_messages realsense2_camera_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/IMUInfo.msg" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_eus _realsense2_camera_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Extrinsics.msg" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_eus _realsense2_camera_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Metadata.msg" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_eus _realsense2_camera_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/srv/DeviceInfo.srv" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_eus _realsense2_camera_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(realsense2_camera_geneus)
add_dependencies(realsense2_camera_geneus realsense2_camera_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS realsense2_camera_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages
_generate_msg_lisp(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/IMUInfo.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/realsense2_camera
)
_generate_msg_lisp(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Extrinsics.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/realsense2_camera
)
_generate_msg_lisp(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Metadata.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/realsense2_camera
)

### Generating Services
_generate_srv_lisp(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/srv/DeviceInfo.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/realsense2_camera
)

### Generating Module File
_generate_module_lisp(realsense2_camera
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/realsense2_camera
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(realsense2_camera_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(realsense2_camera_generate_messages realsense2_camera_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/IMUInfo.msg" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_lisp _realsense2_camera_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Extrinsics.msg" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_lisp _realsense2_camera_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Metadata.msg" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_lisp _realsense2_camera_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/srv/DeviceInfo.srv" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_lisp _realsense2_camera_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(realsense2_camera_genlisp)
add_dependencies(realsense2_camera_genlisp realsense2_camera_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS realsense2_camera_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages
_generate_msg_nodejs(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/IMUInfo.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/realsense2_camera
)
_generate_msg_nodejs(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Extrinsics.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/realsense2_camera
)
_generate_msg_nodejs(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Metadata.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/realsense2_camera
)

### Generating Services
_generate_srv_nodejs(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/srv/DeviceInfo.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/realsense2_camera
)

### Generating Module File
_generate_module_nodejs(realsense2_camera
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/realsense2_camera
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(realsense2_camera_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(realsense2_camera_generate_messages realsense2_camera_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/IMUInfo.msg" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_nodejs _realsense2_camera_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Extrinsics.msg" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_nodejs _realsense2_camera_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Metadata.msg" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_nodejs _realsense2_camera_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/srv/DeviceInfo.srv" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_nodejs _realsense2_camera_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(realsense2_camera_gennodejs)
add_dependencies(realsense2_camera_gennodejs realsense2_camera_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS realsense2_camera_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages
_generate_msg_py(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/IMUInfo.msg"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/realsense2_camera
)
_generate_msg_py(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Extrinsics.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/realsense2_camera
)
_generate_msg_py(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Metadata.msg"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/realsense2_camera
)

### Generating Services
_generate_srv_py(realsense2_camera
  "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/srv/DeviceInfo.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/realsense2_camera
)

### Generating Module File
_generate_module_py(realsense2_camera
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/realsense2_camera
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(realsense2_camera_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(realsense2_camera_generate_messages realsense2_camera_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/IMUInfo.msg" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_py _realsense2_camera_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Extrinsics.msg" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_py _realsense2_camera_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/msg/Metadata.msg" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_py _realsense2_camera_generate_messages_check_deps_${_filename})
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/realsense-ros/realsense2_camera/srv/DeviceInfo.srv" NAME_WE)
add_dependencies(realsense2_camera_generate_messages_py _realsense2_camera_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(realsense2_camera_genpy)
add_dependencies(realsense2_camera_genpy realsense2_camera_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS realsense2_camera_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/realsense2_camera)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/realsense2_camera
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(realsense2_camera_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(realsense2_camera_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/realsense2_camera)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/realsense2_camera
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(realsense2_camera_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(realsense2_camera_generate_messages_eus std_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/realsense2_camera)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/realsense2_camera
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(realsense2_camera_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(realsense2_camera_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/realsense2_camera)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/realsense2_camera
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(realsense2_camera_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(realsense2_camera_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/realsense2_camera)
  install(CODE "execute_process(COMMAND \"/home/zxh/miniconda3/envs/ros1/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/realsense2_camera\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/realsense2_camera
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(realsense2_camera_generate_messages_py sensor_msgs_generate_messages_py)
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(realsense2_camera_generate_messages_py std_msgs_generate_messages_py)
endif()
