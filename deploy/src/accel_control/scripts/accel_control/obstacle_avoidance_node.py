import time

import numpy as np
import cv2
import torch
import onnxruntime as ort

import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Quaternion
from cv_bridge import CvBridge, CvBridgeError

from accel_control.position_control_node import PositionControlNode, lerp, normalize
from accel_control.srv import DepthImage, DepthImageRequest, DepthImageResponse

class ObstacleAvoidanceNode(PositionControlNode):
    def __init__(
        self,
        freq: int,
        model_path: str,
        home_x: float,
        home_y: float,
        target_x: float,
        target_y: float,
        height: float,
        max_acc: float,
        max_vel: float,
        max_dist: float,
        img_height: int,
        img_width: int,
        hover_thrust: float,
        device: torch.device
    ):
        super().__init__(
            freq=freq,
            model_path=model_path,
            home_x=home_x,
            home_y=home_y,
            target_x=target_x,
            target_y=target_y,
            height=height,
            max_acc=max_acc,
            max_vel=max_vel,
            hover_thrust=hover_thrust,
            device=device
        )
        self.max_dist = max_dist
        self.height = img_height
        self.width = img_width
        self.cv_bridge = CvBridge()
        self.gazebo_depth_sub = rospy.Subscriber("/iris_depth_camera/camera/depth/image_raw", Image, self.gazebo_depth_cb, tcp_nodelay=True)
        self.realsense_depth_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.realsense_depth_cb, tcp_nodelay=True)
        self._depth_image = np.ones((self.height, self.width), dtype=np.float32) * max_dist
        self.downsample_interpolation = cv2.INTER_LINEAR
        # self.depth_client = rospy.ServiceProxy("/camera/get_depth_image", DepthImage)
        # self.depth_client.wait_for_service()
    
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
        self.display_depth_image(img)
    
    def realsense_depth_cb(self, msg: Image):
        assert msg.encoding == "16UC1"
        img = self.imgmsg_to_array(msg) * 0.001
        self._depth_image = img
        self.display_depth_image(img)
    
    def get_depth_image(self) -> DepthImageResponse:
        """make majority of processes happen only when the service is called"""
        img = np.nan_to_num(self._depth_image, nan=self.max_dist)
        img = np.array(cv2.resize(img, (self.width, self.height), interpolation=self.downsample_interpolation))
        img = 1 - np.clip(img / self.max_dist, 0., 1.)
        return img
    
    def load_actor(self):
        if self.model_path.endswith(".onnx"):
            self.load_actor_onnx()
        elif self.model_path.endswith(".pt") or self.model_path.endswith(".pt2"):
            self.load_actor_jit()
        else:
            raise ValueError("Unsupported model format. Please use .onnx or .pt/.pt2.")
        self.actor_loaded = True
    
    def inference(self, state: np.ndarray, perception: np.ndarray, orientation: np.ndarray):
        if self.model_path.endswith(".onnx"):
            return self.inference_onnx(state, perception, orientation)
        elif self.model_path.endswith(".pt") or self.model_path.endswith(".pt2"):
            return self.inference_jit(state, perception, orientation)
        self.actor_loaded = True
    
    def load_actor_onnx(self):
        self.actor = ort.InferenceSession(
            self.model_path,
            providers=[("CUDA" if "cuda" in str(self.device) else "CPU") + "ExecutionProvider"],
        )
        input_names = [input.name for input in self.actor.get_inputs()]
        self.need_hidden = "hidden_in" in input_names
        warmup_inputs = {
            input.name: np.random.rand(*input.shape).astype(np.float32) for input in self.actor.get_inputs()}
        if self.need_hidden:
            self.hidden = np.zeros(warmup_inputs["hidden_in"].shape, dtype=np.float32)
        for _ in range(10):
            self.actor.run(None, warmup_inputs)
        rospy.loginfo("Actor warmed up.")
    
    def inference_onnx(self, state: np.ndarray, perception: np.ndarray, orientation: np.ndarray):
        named_inputs = {
            "state": state.astype(np.float32),
            "perception": perception.astype(np.float32),
            "orientation": orientation.astype(np.float32),
            "min_action": self.min_action.astype(np.float32),
            "max_action": self.max_action.astype(np.float32)}
        if self.need_hidden:
            named_inputs["hidden_in"] = self.hidden.astype(np.float32)
            action, quat_xyzw_cmd, acc_norm, hidden = self.actor.run(None, named_inputs)
            self.hidden = hidden
        else:
            action, quat_xyzw_cmd, acc_norm = self.actor.run(None, named_inputs)
        return action.squeeze(0), quat_xyzw_cmd.squeeze(0), acc_norm.squeeze(0)
        
    def load_actor_jit(self):
        rospy.loginfo("Loading actor...")
        self.actor = torch.jit.load(self.model_path, map_location=self.device).to(self.dtype)
        rospy.loginfo("Actor loaded, warming up...")
        self.actor.eval()
        self.need_hidden = self.actor.is_recurrent
        warmup_input = [
            torch.rand(1, 10, **self.factory_kwargs),
            torch.rand(1, self.height, self.width, **self.factory_kwargs),
            torch.rand(1, 3, **self.factory_kwargs),
            torch.rand(1, 3, **self.factory_kwargs),
            torch.rand(1, 3, **self.factory_kwargs)]
        if self.need_hidden:
            warmup_input.append(torch.rand(self.actor.hidden_shape, **self.factory_kwargs))
            self.hidden = torch.zeros(self.actor.hidden_shape, **self.factory_kwargs)
        for _ in range(10):
            self.actor(*warmup_input)
        rospy.loginfo("Actor warmed up.")
    
    def inference_jit(self, state: np.ndarray, perception: np.ndarray, orientation: np.ndarray):
        state = torch.from_numpy(state).to(**self.factory_kwargs)
        perception = torch.from_numpy(perception).to(**self.factory_kwargs)
        orientation = torch.from_numpy(orientation).to(**self.factory_kwargs)
        min_action = torch.from_numpy(self.min_action).to(**self.factory_kwargs)
        max_action = torch.from_numpy(self.max_action).to(**self.factory_kwargs)
        with torch.no_grad():
            if self.need_hidden:
                action, quat_xyzw_cmd, acc_norm, hidden = self.actor(state, perception, orientation, min_action, max_action, self.hidden)
                self.hidden = hidden
            else:
                action, quat_xyzw_cmd, acc_norm = self.actor(state, perception, orientation, min_action, max_action)
        action = action.squeeze(0).cpu().numpy()
        quat_xyzw_cmd = quat_xyzw_cmd.squeeze(0).cpu().numpy()
        acc_norm = acc_norm.squeeze(0).cpu().numpy()
        return action, quat_xyzw_cmd, acc_norm
    
    def step(self):
        tic = time.time()
        target_vel, quat_xyzw, vel, perception = self.get_state()
        self.vel_ema = lerp(self.vel_ema, vel.reshape(1, 3), 0.1)
        state = np.concatenate([target_vel, quat_xyzw, vel], axis=-1).reshape(1, -1)
        target_direction = normalize(target_vel, axis=-1)
        forward = normalize(self.vel_ema, axis=-1)
        zero_yaw = np.array([[1., 0., 0.]])
        # XXX
        orientation = forward
        # orientation = lerp(forward, target_direction, 0.8)
        # orientation = lerp(forward, zero_yaw, np.exp(-np.linalg.norm(target_vel/5)))
        # orientation = self.vel_ema if np.linalg.norm(target_vel) > 1.5 else zero_yaw # hard yaw switch
        
        action, quat_xyzw_cmd, acc_norm = self.inference(state, perception, orientation)
        
        self.acc_cmd = Point(x=action[0], y=action[1], z=action[2])
        thrust = min(1, max(0, acc_norm * self.thrust_factor))
        x, y, z, w = quat_xyzw_cmd
        terminated = self.command_pose(Quaternion(x=x, y=y, z=z, w=w), thrust)
        return time.time() - tic, terminated
    
    def acc_ctrl_cb(self, event: rospy.timer.TimerEvent):
        if self.actor_loaded:
            self.offboard_setpoint_counter += 1
        self.update_target()
        if self.offboard_setpoint_counter == 5*self.freq:
            if not (self.set_mode("OFFBOARD") and self.arm()):
                rospy.logfatal("Failed to switch to OFFBOARD mode and arm the vehicle.")
                exit(1)
        
        if self.t < 20:
            self.set_pos(self.home)
        elif self.t < 110:
            self.set_pose(quat=self.pose_cmd, thrust=self.thrust_cmd)
        elif self.t < 120:
            self.set_pos(self.home)
        else:
            self.terminated = True
    
    def update_target(self):
        if 20 <= self.t < 65:
            self.target = self.target
        elif 65 <= self.t <= 110:
            self.target = self.home
    
    def get_state(self):
        target_vel, quat_xyzw, vel = super().get_state()
        # response: DepthImageResponse = self.depth_client.call(DepthImageRequest(
        #     downsample=True, post_process=True))
        # perception = np.array(self.cv_bridge.imgmsg_to_cv2(response.img)).astype(np.float32)
        perception = self.get_depth_image()
        return target_vel, quat_xyzw, vel, np.expand_dims(perception, axis=0)