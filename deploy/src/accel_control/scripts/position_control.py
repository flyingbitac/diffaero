#! /home/zxh/miniconda3/envs/ros1/bin/python 

import os
import sys
import datetime
import time

import numpy as np
import torch
import torch.nn.functional as F
from line_profiler import LineProfiler

import rospy
from geometry_msgs.msg import Point, Quaternion

sys.path.insert(0, os.path.abspath(".") + "/src/flight_control/scripts")
sys.path.insert(0, os.path.abspath(".") + "/src/accel_control/scripts")
from FlightControl import FlightControlNode
from logger import Logger

class AccelControlNode(FlightControlNode):
    def __init__(
        self,
        freq: int,
        model_path: str,
        max_acc: float,
        hover_thrust: float,
        device: torch.device
    ):
        super(AccelControlNode, self).__init__()
        self.offboard_setpoint_counter = 0
        self.freq = freq
        self.home = Point(x=0., y=0., z=4.) # XXX: replace this with the actual start position
        self.target = Point(x=15., y=15., z=10.) # XXX: replace this with the actual target position
        self.max_vel = 5 # XXX: replace this with the actual max velocity
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
        
        rospy.loginfo("Loading actor...")
        self.actor = torch.jit.load(model_path, map_location=device)
        rospy.loginfo("Actor loaded, warming up...")
        self.actor.eval()
        warmup_input = [
            torch.rand(1, 10, device=device),
            torch.rand(1, 3, device=device),
            torch.rand(1, 3, device=device),
            torch.rand(1, 3, device=device)]
        if self.actor.is_recurrent:
            warmup_input.append(torch.rand(self.actor.hidden_shape, device=device))
        for _ in range(10):
            self.actor(*warmup_input)
        self.hidden = torch.zeros(self.actor.hidden_shape, device=device) if self.actor.is_recurrent else None
        rospy.loginfo("Actor warmed up.")
    
    def step(self):
        tic = time.time()
        target_vel, quat_xyzw, vel = self.get_state()
        self.vel_ema.lerp_(vel.to(device).unsqueeze(0), 0.2)
        state = torch.cat([target_vel, quat_xyzw, vel], dim=-1).to(device).unsqueeze(0)
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
        self.update_target()
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
            self.target.z - self.pos.z])
        target_vel = target_relpos / max(target_relpos.norm(dim=-1).item() / self.max_vel, 1)
        quat_xyzw = torch.tensor([
            self.quat_xyzw.x,
            self.quat_xyzw.y,
            self.quat_xyzw.z,
            self.quat_xyzw.w])
        vel = torch.tensor([
            self.vel.x,
            self.vel.y,
            self.vel.z])
        return target_vel, quat_xyzw, vel

hover_thrust = 0.707 # XXX replace this with actual hover thrust
max_acc = 6. # XXX
control_freq = 50 # Hz
device = torch.device('cpu')
path = "/home/zxh/ws/quaddif/outputs/latest" # XXX
checkpoint_path = os.path.join(path, "checkpoints", "exported_actor.pt2")

@torch.no_grad()
def main():
    rospy.init_node('accel_ctrl', anonymous=True)
    profiler = LineProfiler()
    profiler.add_function(AccelControlNode.step)
    logger = Logger()
    
    node = AccelControlNode(
        freq=control_freq,
        model_path=checkpoint_path,
        max_acc=max_acc,
        hover_thrust=hover_thrust,
        device=device)
    rate = rospy.Rate(control_freq)
    
    while not rospy.is_shutdown():
        inference_time, terminated = node.step()
        logger.log(node)
        if node.offboard_setpoint_counter % (node.freq//5) == 0:
            print(f"|time= {node.offboard_setpoint_counter / node.freq:6.2f}", end=" |")
            print("pos= " + " ".join(map(lambda x: f"{x:5.2f}", [node.pos.x, node.pos.y, node.pos.z])), end=" |")
            print("vel= " + " ".join(map(lambda x: f"{x:5.2f}", [node.vel.x, node.vel.y, node.vel.z])), end=" |")
            print("action= " + " ".join(map(lambda x: f"{x:5.2f}", [node.acc_cmd.x, node.acc_cmd.y, node.acc_cmd.z])), end=" |")
            # print("acc= " + " ".join(map(lambda x: f"{x:5.2f}", [node.acc.x, node.acc.y, node.acc.z]))", end=" |")
            print(f"thrust= {node.thrust_cmd:.2f} |inference_time= {inference_time*1000:6.2f}.ms |")
            if inference_time > 1/control_freq:
                rospy.logwarn("Inference time exceeds control period.")
        if terminated:
            now = datetime.datetime.now()
            path = os.path.join("./outputs/", now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))
            os.makedirs(path, exist_ok=True)
            logger.save_and_plot(path)
            with open(os.path.join(path, "runtime_profile.txt"), "w", encoding="utf-8") as f:
                profiler.print_stats(stream=f, output_unit=1e-3)
            break
        rate.sleep()

if __name__ == "__main__":
    main()