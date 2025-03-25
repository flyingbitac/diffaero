#! /home/zxh/miniconda3/envs/ros1/bin/python 
# coding:utf-8

import numpy as np

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import (
    Point,
    Vector3,
    Quaternion,
    PoseStamped
)
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from mavros_msgs.msg import State, AttitudeTarget
from sensor_msgs.msg import Imu

class FlightControlNode:
    def __init__(self):
        self.pos = Point()
        self.vel = Vector3()
        self.acc = Vector3()
        self.quat_xyzw = Quaternion()
        self.euler = np.zeros(3)
        
        self.current_state = State()
        
        self.state_sub = rospy.Subscriber("mavros/state", State, self.state_cb, tcp_nodelay=True)
        # append xml code below to PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris/iris.sdf:
        # </plugin>
        #     <plugin name="p3d_base_controller" filename="libgazebo_ros_p3d.so">
        #     <alwaysOn>true</alwaysOn>
        #     <updateRate>50.0</updateRate>
        #     <bodyName>base_link</bodyName>
        #     <topicName>some_object_name_vrpn_client/estimated_odometry</topicName>
        #     <gaussianNoise>0.01</gaussianNoise>
        #     <frameName>world</frameName>
        #     <xyzOffsets>0 0 0</xyzOffsets>
        #     <rpyOffsets>0 0 0</rpyOffsets>
        # </plugin>
        self.odom_sub = rospy.Subscriber("some_object_name_vrpn_client/estimated_odometry", Odometry, self.odom_cb, tcp_nodelay=True)
        # self.odom_sub = rospy.Subscriber("mavros/odometry/in", Odometry, self.odom_cb, tcp_nodelay=True)
        self.acc_sub = rospy.Subscriber("mavros/imu/data", Imu, self.acc_cb, tcp_nodelay=True)
        
        self.pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.pose_pub = rospy.Publisher("mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)
        
        self.arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
        self.set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
    
    def state_cb(self, msg: State):
        self.current_state = msg
    
    def odom_cb(self, msg: Odometry):
        self.pos = msg.pose.pose.position
        self.vel = msg.twist.twist.linear
        self.quat_xyzw = msg.pose.pose.orientation
        x, y, z, w = self.quat_xyzw.x, self.quat_xyzw.y, self.quat_xyzw.z, self.quat_xyzw.w
        roll = np.arctan2(2.0 * (w * x - y * z), 1.0 - 2.0 * (x**2 + y**2))
        pitch = np.arcsin(2.0 * (w * y + x * z))
        yaw = np.arctan2(2.0 * (w * z - x * y), 1.0 - 2.0 * (y**2 + z**2))
        self.euler = np.array([roll, pitch, yaw])
    
    def acc_cb(self, msg: Imu):
        self.acc = msg.linear_acceleration
    
    def set_pos(self, pos: Point):
        msg = PoseStamped()
        msg.pose.position = pos
        msg.header.stamp = rospy.Time.now()
        self.pos_pub.publish(msg)
    
    def set_pose(self, quat: Quaternion, thrust: float):
        pose_msg = AttitudeTarget()
        pose_msg.orientation = quat
        pose_msg.thrust = thrust
        pose_msg.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE +
            AttitudeTarget.IGNORE_PITCH_RATE +
            AttitudeTarget.IGNORE_YAW_RATE)
        pose_msg.header.stamp = rospy.Time.now()
        self.pose_pub.publish(pose_msg)
    
    def arm(self):
        try:
            arm_cmd = CommandBoolRequest()
            arm_cmd.value = True
            response = self.arming_client(arm_cmd)
            rospy.loginfo("Vehicle Armed." if response.success else "Vehicle Arm Failed.")
            return response.success
        except rospy.ServiceException as e:
            rospy.logerr("Arming service call failed: %s", e)
            return False
    
    def disarm(self):
        try:
            arm_cmd = CommandBoolRequest()
            arm_cmd.value = True
            response = self.arming_client(arm_cmd)
            rospy.loginfo("Vehicle Disarmed." if response.success else "Vehicle Disarm Failed.")
            return response.success
        except rospy.ServiceException as e:
            rospy.logerr("Disarming service call failed: %s", e)
            return False
    
    def set_mode(self, mode="OFFBOARD"):
        try:
            offb_set_mode = SetModeRequest()
            offb_set_mode.custom_mode = 'OFFBOARD'
            response = self.set_mode_client(offb_set_mode)
            rospy.loginfo(f"Vehicle mode set to {mode}." if response.mode_sent else f"Failed to set vehicle mode to {mode}.")
            return response.mode_sent
        except rospy.ServiceException as e:
            rospy.logerr("Set mode service call failed: %s", e)
            return False
