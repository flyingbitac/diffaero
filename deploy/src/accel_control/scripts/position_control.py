#! /home/zxh/miniconda3/envs/ros1/bin/python 

import os
import sys
import datetime
import time
import math

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from line_profiler import LineProfiler

import rospy
from geometry_msgs.msg import Point, Quaternion

sys.path.insert(0, os.path.abspath(".") + "/src/flight_control/scripts")
from FlightControl import FlightControlNode

class AccelControlNode(FlightControlNode):
    def __init__(self, freq: int):
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
    
    def acc_ctrl_cb(self, event: rospy.timer.TimerEvent):
        self.offboard_setpoint_counter += 1
        if self.offboard_setpoint_counter == 5*self.freq:
            if not (self.set_mode("OFFBOARD") and self.arm()):
                rospy.logfatal("Failed to switch to OFFBOARD mode and arm the vehicle.")
                exit(1)
        
        t = self.offboard_setpoint_counter / self.freq
        # self.target = Point(
        #     x=2 * math.sin(1.4 * t),
        #     y=2 * math.cos(1.4 * t),
        #     z=10)
        L = 7
        if (int(t) - 20) % 20 <= 5:
            self.target = Point(-L, L, 10)
        elif (int(t) - 20) % 20 <= 10:
            self.target = Point(L, L, 10)
        elif (int(t) - 20) % 20 <= 15:
            self.target = Point(L, -L, 10)
        elif (int(t) - 20) % 20 <= 20:
            self.target = Point(-L, -L, 10)
        
        if t < 20:
            self.set_pos(self.home)
        elif t < 60:
            self.set_pose(quat=self.pose_cmd, thrust=self.thrust_cmd)
        elif t < 70:
            self.set_pos(self.home)
        else:
            self.terminated = True
    
    def command_pose(self, quat: Quaternion, thrust: float):
        self.pose_cmd = quat
        self.thrust_cmd = thrust
        return self.terminated
    
    def get_state(self):
        target_relpos = torch.tensor([
            self.target.x - self.pos.x,
            self.target.y - self.pos.y,
            self.target.z - self.pos.z
        ])
        target_vel = target_relpos / max(target_relpos.norm(dim=-1).item() / self.max_vel, 1)
        quat_xyzw = torch.tensor([
            self.quat_xyzw.x,
            self.quat_xyzw.y,
            self.quat_xyzw.z,
            self.quat_xyzw.w,
        ])
        vel = torch.tensor([
            self.vel.x,
            self.vel.y,
            self.vel.z
        ])
        return target_vel, quat_xyzw, vel

class Logger:
    def __init__(self):
        self.ts = []
        self.pos = []
        self.vel = []
        self.acc = []
        self.acc_cmd = []
        self.thrusts = []
        self.euler = []
    
    def log(self, node: AccelControlNode):
        self.ts.append(node.offboard_setpoint_counter/node.freq)
        self.pos.append((node.pos.x, node.pos.y, node.pos.z))
        self.vel.append((node.vel.x, node.vel.y, node.vel.z))
        self.acc.append((node.acc.x, node.acc.y, node.acc.z))
        self.acc_cmd.append((node.acc_cmd.x, node.acc_cmd.y, node.acc_cmd.z))
        self.thrusts.append(node.thrust_cmd)
        self.euler.append(node.euler)
    
    def save_and_plot(self, path: str):
        time = np.array(self.ts)
        pos = np.array(self.pos)
        vel = np.array(self.vel)
        acc = np.array(self.acc)
        acc_cmd = np.array(self.acc_cmd)
        thrusts = np.array(self.thrusts)
        euler = np.array(self.euler) * 180 / math.pi
        df = pd.DataFrame({"time": self.ts,
            "pos_x": pos[:, 0], "pos_y": pos[:, 1], "pos_z": pos[:, 2],
            "vel_x": vel[:, 0], "vel_y": vel[:, 1], "vel_z": vel[:, 2],
            "acc_x": acc[:, 0], "acc_y": acc[:, 1], "acc_z": acc[:, 2],
            "acc_cmd_x": acc_cmd[:, 0], "acc_cmd_y": acc_cmd[:, 1], "acc_cmd_z": acc_cmd[:, 2],
            "roll": euler[:, 0], "pitch": euler[:, 1], "yaw": euler[:, 2],
            "thrust": thrusts
        })
        
        df.to_csv(os.path.join(path, "data.csv"), index=False)
        
        for (data, name) in zip(
            [pos, vel, acc, acc_cmd, euler],
            ["pos", "vel", "acc_imu", "acc_cmd", "euler angles"]
        ):
            fig = plt.figure(dpi=200)
            plt.suptitle(name)
            for i in range(3):
                plt.subplot(3, 1, i+1)
                plt.plot(time, data[:, i])
                plt.xlabel("time(s)")
                if name == "euler":
                    plt.ylabel(["roll", "pitch", "yaw"][i] + "(deg)")
                else:
                    plt.ylabel("xyz"[i]+" axis")
                plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(path, name+".png"))
        
        fig = plt.figure(dpi=200)
        plt.title("thrust command")
        plt.plot(time, thrusts)
        plt.ylim((-0.1, 1.1))
        plt.xlabel("time(s)")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(path, "thrust.png"))
        
        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(111, projection='3d')
        plt.title("3d trajectory")
        plt.plot(pos[:, 0], pos[:, 1], pos[:, 2])
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(path, "3d_trajectory.png"))

hover_thrust = 0.707 # XXX replace this with actual hover thrust
thrust_factor = hover_thrust / 9.81
max_acc = 6. # XXX
control_freq = 50 # Hz
device = torch.device('cpu')
path = "/home/zxh/ws/quaddif/outputs/2025-01-07/15-55-03"
checkpoint_path = os.path.join(path, "checkpoints", "exported_actor.pt2")
cfg_path = os.path.join(path, ".hydra", "config.yaml")
min_action = torch.tensor([[-max_acc, -max_acc, 0]], device=device)
max_action = torch.tensor([[max_acc, max_acc, 40]], device=device)

@torch.no_grad()
def main():
    rospy.init_node('accel_ctrl', anonymous=True)
    profiler = LineProfiler()
    logger = Logger()
    vel_ema = torch.zeros(1, 3, device=device)
    
    rospy.loginfo("Loading actor...")
    actor = torch.jit.load(checkpoint_path, map_location=device)
    rospy.loginfo("Actor loaded, warming up...")
    actor.eval()
    fake_input = torch.rand(10, device=device)
    for _ in range(10):
        actor(fake_input)
    actor.reset()
    rospy.loginfo("Actor warmed up.")
    
    node = AccelControlNode(freq=control_freq)
    rate = rospy.Rate(control_freq)
    
    @profiler
    def step():
        tic = time.time()
        target_vel, quat_xyzw, vel = node.get_state()
        vel_ema.lerp_(vel.to(device).unsqueeze(0), 0.2)
        state = torch.cat([target_vel, quat_xyzw, vel], dim=-1).to(device)
        raw_action: torch.Tensor = actor(state)
        action = actor.rescale(raw_action, min_action, max_action)
        node.acc_cmd = Point(x=action[0, 0].item(), y=action[0, 1].item(), z=action[0, 2].item())
        # XXX
        orientation = F.normalize(vel_ema, dim=-1)
        # orientation = F.normalize(vel_ema, dim=-1).lerp(torch.tensor([[1., 0., 0.]]), target_vel.norm(dim=-1).neg().exp().item()) # soft yaw switch
        # orientation = vel_ema if target_vel.norm(dim=-1).item() > 1.5 else torch.tensor([[1., 0., 0.]]) # hard yaw switch
        quat_xyzw_cmd, acc_norm = actor.post_process(action, orientation)
        thrust = acc_norm.mul(thrust_factor).clamp(0, 1).item()
        x, y, z, w = list(map(lambda x: x.item(), quat_xyzw_cmd[0].unbind(dim=-1)))
        terminated = node.command_pose(Quaternion(x=x, y=y, z=z, w=w), thrust)
        logger.log(node)
        return time.time() - tic, terminated, action, thrust
    
    while not rospy.is_shutdown():
        inference_time, terminated, action, thrust = step()
        if node.offboard_setpoint_counter % (node.freq//5) == 0:
            log_str = (
                f"|time= {node.offboard_setpoint_counter / node.freq:6.2f} |" +
                "pos= " + " ".join(map(lambda x: f"{x:5.2f}", [node.pos.x, node.pos.y, node.pos.z])) + " |" +
                "vel= " + " ".join(map(lambda x: f"{x:5.2f}", [node.vel.x, node.vel.y, node.vel.z])) + " |" +
                # "acc= " + " ".join(map(lambda x: f"{x:5.2f}", [node.acc.x, node.acc.y, node.acc.z])) + " |" +
                "action= " + " ".join(map(lambda x: f"{x.item():5.2f}", action.unbind(dim=-1))) + " |" +
                f"thrust= {thrust:.2f} |inference_time= {inference_time*1000:6.2f}.ms |"
            )
            rospy.loginfo(log_str)
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