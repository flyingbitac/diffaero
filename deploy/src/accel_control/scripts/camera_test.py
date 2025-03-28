import cv2
import rospy
import torch
import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from accel_control.camera_node import CameraNode
from accel_control.srv import DepthImage, DepthImageRequest, DepthImageResponse

def main():
    max_dist = 6.
    rospy.init_node('obstacle_avoidance_node', anonymous=True)
    node = CameraNode(max_dist, 9, 16)
    client = rospy.ServiceProxy("/camera/get_depth_image", DepthImage)
    client.wait_for_service()
    device = torch.device("cuda")
    cv_bridge = CvBridge()
    
    def callback(event: rospy.timer.TimerEvent):
        print("CALLBACK CALLED")
        response: DepthImageResponse = client.call(DepthImageRequest(
            downsample=True,
            post_process=True
        ))
        img: np.ndarray = cv_bridge.imgmsg_to_cv2(response.img)
        cv2.imshow("depth", img)
        cv2.waitKey(1)
    
    timer = rospy.Timer(rospy.Duration(1/30), callback)
    rospy.spin()

if __name__ == "__main__":
    main()