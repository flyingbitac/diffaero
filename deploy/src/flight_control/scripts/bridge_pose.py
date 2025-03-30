#! /usr/bin/env python
from functools import partial

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

def bridge_pose(pose_pub: rospy.Publisher, data: Odometry):
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "optitrack"
    # pos.child_frame_id = "base_link"
    pose.pose = data.pose.pose
    # rospy.loginfo("pub pos data: Position: %s, Orientation: %s",pos.pose.position, pos.pose.orientation)

    pose_pub.publish(pose)

if __name__ == '__main__':
    rospy.init_node('bridge_pose')
    pose_pub = rospy.Publisher('/mavros/vision_pose/pose', PoseStamped, queue_size=1)
    vrpn_sub = rospy.Subscriber('/some_object_name_vrpn_client/estimated_odometry', Odometry, partial(bridge_pose, pose_pub))
    rospy.spin()