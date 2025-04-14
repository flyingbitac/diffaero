# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "accel_control: 0 messages, 1 services")

set(MSG_I_FLAGS "-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(accel_control_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/accel_control/srv/DepthImage.srv" NAME_WE)
add_custom_target(_accel_control_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "accel_control" "/home/zxh/ws/wrqws/quaddif/deploy/src/accel_control/srv/DepthImage.srv" "sensor_msgs/Image:std_msgs/Header"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(accel_control
  "/home/zxh/ws/wrqws/quaddif/deploy/src/accel_control/srv/DepthImage.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/accel_control
)

### Generating Module File
_generate_module_cpp(accel_control
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/accel_control
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(accel_control_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(accel_control_generate_messages accel_control_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/accel_control/srv/DepthImage.srv" NAME_WE)
add_dependencies(accel_control_generate_messages_cpp _accel_control_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(accel_control_gencpp)
add_dependencies(accel_control_gencpp accel_control_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS accel_control_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(accel_control
  "/home/zxh/ws/wrqws/quaddif/deploy/src/accel_control/srv/DepthImage.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/accel_control
)

### Generating Module File
_generate_module_eus(accel_control
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/accel_control
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(accel_control_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(accel_control_generate_messages accel_control_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/accel_control/srv/DepthImage.srv" NAME_WE)
add_dependencies(accel_control_generate_messages_eus _accel_control_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(accel_control_geneus)
add_dependencies(accel_control_geneus accel_control_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS accel_control_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(accel_control
  "/home/zxh/ws/wrqws/quaddif/deploy/src/accel_control/srv/DepthImage.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/accel_control
)

### Generating Module File
_generate_module_lisp(accel_control
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/accel_control
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(accel_control_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(accel_control_generate_messages accel_control_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/accel_control/srv/DepthImage.srv" NAME_WE)
add_dependencies(accel_control_generate_messages_lisp _accel_control_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(accel_control_genlisp)
add_dependencies(accel_control_genlisp accel_control_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS accel_control_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(accel_control
  "/home/zxh/ws/wrqws/quaddif/deploy/src/accel_control/srv/DepthImage.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/accel_control
)

### Generating Module File
_generate_module_nodejs(accel_control
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/accel_control
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(accel_control_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(accel_control_generate_messages accel_control_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/accel_control/srv/DepthImage.srv" NAME_WE)
add_dependencies(accel_control_generate_messages_nodejs _accel_control_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(accel_control_gennodejs)
add_dependencies(accel_control_gennodejs accel_control_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS accel_control_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(accel_control
  "/home/zxh/ws/wrqws/quaddif/deploy/src/accel_control/srv/DepthImage.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/noetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/noetic/share/std_msgs/cmake/../msg/Header.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/accel_control
)

### Generating Module File
_generate_module_py(accel_control
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/accel_control
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(accel_control_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(accel_control_generate_messages accel_control_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/zxh/ws/wrqws/quaddif/deploy/src/accel_control/srv/DepthImage.srv" NAME_WE)
add_dependencies(accel_control_generate_messages_py _accel_control_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(accel_control_genpy)
add_dependencies(accel_control_genpy accel_control_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS accel_control_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/accel_control)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/accel_control
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(accel_control_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(accel_control_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/accel_control)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/accel_control
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(accel_control_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(accel_control_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/accel_control)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/accel_control
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(accel_control_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(accel_control_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/accel_control)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/accel_control
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(accel_control_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(accel_control_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/accel_control)
  install(CODE "execute_process(COMMAND \"/home/zxh/miniconda3/envs/ros1/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/accel_control\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/accel_control
    DESTINATION ${genpy_INSTALL_DIR}
    # skip all init files
    PATTERN "__init__.py" EXCLUDE
    PATTERN "__init__.pyc" EXCLUDE
  )
  # install init files which are not in the root folder of the generated code
  string(REGEX REPLACE "([][+.*()^])" "\\\\\\1" ESCAPED_PATH "${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/accel_control")
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/accel_control
    DESTINATION ${genpy_INSTALL_DIR}
    FILES_MATCHING
    REGEX "${ESCAPED_PATH}/.+/__init__.pyc?$"
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(accel_control_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(accel_control_generate_messages_py sensor_msgs_generate_messages_py)
endif()
