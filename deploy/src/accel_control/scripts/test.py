import rospy

from accel_control.test_node import TestNode

if __name__ == "__main__":
    rospy.init_node('obstacle_avoidance_node', anonymous=True)
    node = TestNode(10, 9, 16)
    rospy.spin()