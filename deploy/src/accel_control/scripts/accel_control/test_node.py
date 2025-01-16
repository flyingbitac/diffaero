import cv2
import numpy as np
import torch
import torch.nn.functional as F

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class TestNode:
    def __init__(
        self,
        max_dist: float,
        height: int,
        width: int
    ):
        self.CvImg = CvBridge()
        self.depth_sub = rospy.Subscriber("/iris_depth_camera/camera/depth/image_raw", Image, self.depth_cb, tcp_nodelay=True)
        self.max_dist = max_dist
        self.height = height
        self.width = width
    
    def depth_cb(self, msg: Image):
        img = torch.from_numpy(self.CvImg.imgmsg_to_cv2(msg, "32FC1"))
        img.nan_to_num_(nan=self.max_dist).clamp_(max=self.max_dist)
        img.div_(self.max_dist).sub_(1).neg_()
        img = F.adaptive_max_pool2d(img[None, ...], (self.height, self.width)).squeeze(0)
        
        N = 16
        cv2.imshow("depth", cv2.resize(img.cpu().numpy(), (N*16, N*9), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(1)