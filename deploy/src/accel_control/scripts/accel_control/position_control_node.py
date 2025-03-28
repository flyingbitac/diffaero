import time

import torch
import torch.nn.functional as F

import rospy
from geometry_msgs.msg import Point, Quaternion

from flight_control.FlightControl import FlightControlNode

class PositionControlNode(FlightControlNode):
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
        hover_thrust: float,
        device: torch.device
    ):
        super().__init__()
        self.offboard_setpoint_counter = 0
        self.freq = freq
        self.model_path = model_path
        self.home = Point(x=home_x, y=home_y, z=height) # XXX: replace this with the actual start position
        self.target = Point(x=target_x, y=target_y, z=height) # XXX: replace this with the actual target position
        self.max_vel = max_vel
        self.timer = rospy.Timer(rospy.Duration(1/freq), self.acc_ctrl_cb)
        self.acc_cmd = Point()
        self.pose_cmd = Quaternion()
        self.thrust_cmd = 0
        self.terminated = False
        self.thrust_factor = hover_thrust / 9.81
        self.device = device
        
        self.min_action = torch.tensor([[-max_acc, -max_acc, 0]], device=device)
        self.max_action = torch.tensor([[max_acc, max_acc, 40]], device=device)
        self.vel_ema = torch.zeros(1, 3, device=device)
    
    def load_actor(self):
        rospy.loginfo("Loading actor...")
        self.actor = torch.jit.load(self.model_path, map_location=self.device)
        rospy.loginfo("Actor loaded, warming up...")
        self.actor.eval()
        warmup_input = [
            torch.rand(1, 10, device=self.device),
            torch.rand(1, 3, device=self.device),
            torch.rand(1, 3, device=self.device),
            torch.rand(1, 3, device=self.device)]
        if self.actor.is_recurrent:
            warmup_input.append(torch.rand(self.actor.hidden_shape, device=self.device))
            self.hidden = torch.zeros(self.actor.hidden_shape, device=self.device)
        for _ in range(10):
            self.actor(*warmup_input)
        rospy.loginfo("Actor warmed up.")
    
    @torch.no_grad()
    def step(self):
        tic = time.time()
        target_vel, quat_xyzw, vel = self.get_state()
        self.vel_ema.lerp_(vel.to(self.device).unsqueeze(0), 0.2)
        state = torch.cat([target_vel, quat_xyzw, vel], dim=-1).to(self.device).unsqueeze(0)
        forward = F.normalize(self.vel_ema, dim=-1)
        zero_yaw = torch.tensor([[1., 0., 0.]])
        # XXX
        orientation = forward
        # orientation = forward.lerp(zero_yaw, target_vel.norm(dim=-1).neg().exp().item()) # soft yaw switch
        # orientation = self.vel_ema if target_vel.norm(dim=-1).item() > 1.5 else zero_yaw # hard yaw switc
        if self.actor.is_recurrent:
            action, quat_xyzw_cmd, acc_norm, hidden = self.actor(state, orientation, self.min_action, self.max_action, self.hidden)
            self.hidden = hidden
        else:
            action, quat_xyzw_cmd, acc_norm = self.actor(state, orientation, self.min_action, self.max_action)
        self.acc_cmd = Point(x=action[0, 0].item(), y=action[0, 1].item(), z=action[0, 2].item())
        thrust = acc_norm.mul(self.thrust_factor).clamp(0, 1).item()
        x, y, z, w = list(map(lambda x: x.item(), quat_xyzw_cmd[0].unbind(dim=-1)))
        terminated = self.command_pose(Quaternion(x=x, y=y, z=z, w=w), thrust)
        return time.time() - tic, terminated
    
    def acc_ctrl_cb(self, event: rospy.timer.TimerEvent):
        self.offboard_setpoint_counter += 1
        # self.update_target()
        if self.offboard_setpoint_counter == 5*self.freq:
            if not (self.set_mode("OFFBOARD") and self.arm()):
                rospy.logfatal("Failed to switch to OFFBOARD mode and arm the vehicle.")
                exit(1)
                
        t = self.offboard_setpoint_counter / self.freq
        if t < 20:
            self.set_pos(self.home)
        elif t < 60:
            self.set_pose(quat=self.pose_cmd, thrust=self.thrust_cmd)
        elif t < 70:
            self.set_pos(self.home)
        else:
            self.terminated = True
    
    def update_target(self):
        t = self.offboard_setpoint_counter / self.freq
        H = 10
        
        # R = 2
        # self.target = Point(
        #     x=R * math.sin(1.4 * t),
        #     y=R * math.cos(1.4 * t),
        #     z=H)
        
        L = 7
        if (int(t) - 20) % 20 <= 5:
            self.target = Point(-L, L, H)
        elif (int(t) - 20) % 20 <= 10:
            self.target = Point(L, L, H)
        elif (int(t) - 20) % 20 <= 15:
            self.target = Point(L, -L, H)
        elif (int(t) - 20) % 20 <= 20:
            self.target = Point(-L, -L, H)
    
    def command_pose(self, quat: Quaternion, thrust: float):
        self.pose_cmd = quat
        self.thrust_cmd = thrust
        return self.terminated
    
    def get_state(self):
        target_relpos = torch.tensor([
            self.target.x - self.pos.x,
            self.target.y - self.pos.y,
            self.target.z - self.pos.z], device=self.device)
        target_vel = target_relpos / max(target_relpos.norm(dim=-1).item() / self.max_vel, 1)
        quat_xyzw = torch.tensor([
            self.quat_xyzw.x,
            self.quat_xyzw.y,
            self.quat_xyzw.z,
            self.quat_xyzw.w], device=self.device)
        vel = torch.tensor([
            self.vel.x,
            self.vel.y,
            self.vel.z], device=self.device)
        return target_vel, quat_xyzw, vel
