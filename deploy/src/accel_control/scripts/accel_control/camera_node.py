import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image, CameraInfo
from accel_control.srv import DepthImage, DepthImageRequest, DepthImageResponse
from cv_bridge import CvBridge, CvBridgeError

class CameraNode:
    def __init__(
        self,
        max_dist: float,
        height: int,
        width: int,
        display: bool = False
    ):
        self.cv_bridge = CvBridge()
        self.gazebo_depth_sub = rospy.Subscriber("/iris_depth_camera/camera/depth/image_raw", Image, self.gazebo_depth_cb, tcp_nodelay=True)
        self.realsense_depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.realsense_depth_cb, tcp_nodelay=True)
        self.realsense_info_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.realsense_info_cb, tcp_nodelay=True)
        self.depth_srv = rospy.Service("/camera/get_depth_image", DepthImage, self.get_depth_image)
        self.max_dist = max_dist
        self.height = height
        self.width = width
        self.display = display
        self.downsample_interpolation = cv2.INTER_LINEAR # [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC]
        
        self._depth_image = np.ones((height, width), dtype=np.float32) * max_dist
    
    def imgmsg_to_array(self, msg: Image) -> np.ndarray:
        return np.array(self.cv_bridge.imgmsg_to_cv2(msg, "passthrough"), dtype=np.float32)

    def display_depth_image(self, img: np.ndarray):
        img = np.nan_to_num(img, nan=self.max_dist)
        img = np.array(cv2.resize(img, (self.width, self.height), interpolation=self.downsample_interpolation))
        img = np.array(cv2.resize(img, (320, 180), interpolation=cv2.INTER_NEAREST))
        img = 1 - np.clip(img / self.max_dist, 0., 1.)
        cv2.imshow("depth", (img * 255).astype(np.uint8))
        cv2.waitKey(1)
    
    def gazebo_depth_cb(self, msg: Image):
        assert msg.encoding == "32FC1"
        img = self.imgmsg_to_array(msg)
        self._depth_image = img
        if self.display:
            self.display_depth_image(img)
    
    def realsense_depth_cb(self, msg: Image):
        assert msg.encoding == "16UC1"
        img = self.imgmsg_to_array(msg) * 0.001
        self._depth_image = img
        if self.display:
            self.display_depth_image(img)
    
    def get_depth_image(self, request: DepthImageRequest) -> DepthImageResponse:
        """make majority of processes happen only when the service is called"""
        img = np.nan_to_num(self._depth_image, nan=self.max_dist)
        if request.downsample:
            img = np.array(cv2.resize(img, (self.width, self.height), interpolation=self.downsample_interpolation))
        if request.post_process:
            img = 1 - np.clip(img / self.max_dist, 0., 1.)
        imgmsg = self.cv_bridge.cv2_to_imgmsg(img, "passthrough")
        assert imgmsg.encoding == "32FC1"
        response = DepthImageResponse(imgmsg)
        return response
    
    def realsense_info_cb(self, msg: CameraInfo):
        pass
        # print("height and width: ", msg.height, msg.width)