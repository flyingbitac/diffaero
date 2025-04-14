import rospy

from accel_control.camera_node import CameraNode

def main():
    rospy.init_node('obstacle_avoidance_node', anonymous=True)
    node = CameraNode(
        max_dist=rospy.get_param("~max_dist"),
        height=rospy.get_param("~img_height"),
        width=rospy.get_param("~img_width"),
        display=rospy.get_param("~display")
    )
    rospy.spin()

if __name__ == "__main__":
    main()