import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import rospy
from geometry_msgs.msg import Point, Quaternion
from cv_bridge import CvBridge, CvBridgeError

from accel_control.position_control_node import PositionControlNode
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
        self.cv_bridge = CvBridge()
        self.depth_client = rospy.ServiceProxy("/camera/get_depth_image", DepthImage)
        self.depth_client.wait_for_service()
        self.home = Point(x=home_x, y=home_y, z=height) # XXX: replace this with the actual start position
        self.target = Point(x=target_x, y=target_y, z=height) # XXX: replace this with the actual target position

        self.max_dist = max_dist
        self.height = img_height
        self.width = img_width
        
    def load_actor(self):
        rospy.loginfo("Loading actor...")
        self.actor = torch.jit.load(self.model_path, map_location=self.device)
        rospy.loginfo("Actor loaded, warming up...")
        self.actor.eval()
        warmup_input = [
            torch.rand(1, 10, device=self.device),
            torch.rand(1, self.height, self.width, device=self.device),
            torch.rand(1, 3, device=self.device),
            torch.rand(1, 3, device=self.device),
            torch.rand(1, 3, device=self.device)]
        if self.actor.is_recurrent:
            warmup_input.append(torch.rand(self.actor.hidden_shape, device=self.device))
            self.hidden = torch.zeros(self.actor.hidden_shape, device=self.device)
        for _ in range(10):
            self.actor(*warmup_input)
        rospy.loginfo("Actor warmed up.")
    
    def acc_ctrl_cb(self, event: rospy.timer.TimerEvent):
        self.offboard_setpoint_counter += 1
        self.update_target()
        if self.offboard_setpoint_counter == 5*self.freq:
            if not (self.set_mode("OFFBOARD") and self.arm()):
                rospy.logfatal("Failed to switch to OFFBOARD mode and arm the vehicle.")
                exit(1)
                
        t = self.offboard_setpoint_counter / self.freq
        if t < 20:
            self.set_pos(self.home)
        elif t < 110:
            self.set_pose(quat=self.pose_cmd, thrust=self.thrust_cmd)
        elif t < 120:
            self.set_pos(self.home)
        else:
            self.terminated = True
    
    @torch.no_grad()
    def step(self):
        tic = time.time()
        target_vel, quat_xyzw, vel, perception = self.get_state()
        self.vel_ema.lerp_(vel.to(self.device).unsqueeze(0), 0.4)
        state = torch.cat([target_vel, quat_xyzw, vel], dim=-1).to(self.device).unsqueeze(0)
        forward = F.normalize(self.vel_ema, dim=-1)
        zero_yaw = torch.tensor([[1., 0., 0.]], device=self.device)
        # XXX
        orientation = forward
        # orientation = forward.lerp(F.normalize(target_vel, dim=-1), 0.8)
        # orientation = forward.lerp(zero_yaw, target_vel.norm(dim=-1).neg().exp().item()) # soft yaw switch
        # orientation = self.vel_ema if target_vel.norm(dim=-1).item() > 1.5 else zero_yaw # hard yaw switc
        if self.actor.is_recurrent:
            action, quat_xyzw_cmd, acc_norm, hidden = self.actor(state, perception, orientation, self.min_action, self.max_action, self.hidden)
            self.hidden = hidden
        else:
            action, quat_xyzw_cmd, acc_norm = self.actor(state, perception, orientation, self.min_action, self.max_action)
        self.acc_cmd = Point(x=action[0, 0].item(), y=action[0, 1].item(), z=action[0, 2].item())
        thrust = acc_norm.mul(self.thrust_factor).clamp(0, 1).item()
        x, y, z, w = list(map(lambda x: x.item(), quat_xyzw_cmd[0].unbind(dim=-1)))
        terminated = self.command_pose(Quaternion(x=x, y=y, z=z, w=w), thrust)
        return time.time() - tic, terminated
    
    def update_target(self):
        t = self.offboard_setpoint_counter / self.freq
        if 20 <= t < 65:
            self.target = self.target
        elif 65 <= t <= 110:
            self.target = self.home
    
    def get_state(self):
        target_vel, quat_xyzw, vel = super().get_state()
        response: DepthImageResponse = self.depth_client.call(DepthImageRequest(
            downsample=True, post_process=True))
        perception = torch.from_numpy(self.cv_bridge.imgmsg_to_cv2(response.img)).to(self.device)
        return target_vel, quat_xyzw, vel, perception.unsqueeze(0)
