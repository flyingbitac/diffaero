#! /home/zxh/miniconda3/envs/ros1/bin/python 

import os
import sys
import datetime
import math

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import rospy
from geometry_msgs.msg import Point, Quaternion

sys.path.insert(0, os.path.abspath(".") + "/src/flight_control/scripts")
from FlightControl import FlightControlNode

class AccelControlNode(FlightControlNode):
    def __init__(self, freq):
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
        # self.target = Point(
        #     x=5 * math.sin(0.6 * self.offboard_setpoint_counter / self.freq),
        #     y=5 * math.cos(0.6 * self.offboard_setpoint_counter / self.freq),
        #     z=10)
        if self.offboard_setpoint_counter <= 20*self.freq:
            self.set_pos(self.home)
        elif self.offboard_setpoint_counter <= 60*self.freq:
            self.set_pose(quat=self.pose_cmd, thrust=self.thrust_cmd)
        elif self.offboard_setpoint_counter <= 70*self.freq:
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
        acc = torch.tensor([
            self.acc.x,
            self.acc.y,
            self.acc.z
        ])
        return target_vel, quat_xyzw, vel, acc

class Logger:
    def __init__(self):
        self.ts = []
        self.pos = []
        self.vel = []
        self.acc = []
        self.acc_cmd = []
        self.thrusts = []
    
    def log(self, node: AccelControlNode):
        self.ts.append(node.offboard_setpoint_counter/node.freq)
        self.pos.append((node.pos.x, node.pos.y, node.pos.z))
        self.vel.append((node.vel.x, node.vel.y, node.vel.z))
        self.acc.append((node.acc.x, node.acc.y, node.acc.z))
        self.acc_cmd.append((node.acc_cmd.x, node.acc_cmd.y, node.acc_cmd.z))
        self.thrusts.append(node.thrust_cmd)
    
    def save_and_plot(self, path: str):
        time = np.array(self.ts)
        pos = np.array(self.pos)
        vel = np.array(self.vel)
        acc = np.array(self.acc)
        acc_cmd = np.array(self.acc_cmd)
        thrusts = np.array(self.thrusts)
        df = pd.DataFrame({"time": self.ts,
            "pos_x": pos[:, 0], "pos_y": pos[:, 1], "pos_z": pos[:, 2],
            "vel_x": vel[:, 0], "vel_y": vel[:, 1], "vel_z": vel[:, 2],
            "acc_x": acc[:, 0], "acc_y": acc[:, 1], "acc_z": acc[:, 2],
            "acc_cmd_x": acc_cmd[:, 0], "acc_cmd_y": acc_cmd[:, 1], "acc_cmd_z": acc_cmd[:, 2],
            "thrust": thrusts
        })
        
        now = datetime.datetime.now()
        path = os.path.join(path, now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))
        os.makedirs(path, exist_ok=True)
        df.to_csv(os.path.join(path, "data.csv"), index=False)
        
        for (data, name) in zip(
            [pos, vel, acc, acc_cmd],
            ["pos", "vel", "acc_imu", "acc_cmd"]
        ):
            fig = plt.figure(dpi=200)
            plt.suptitle(name)
            for i in range(3):
                plt.subplot(3, 1, i+1)
                plt.plot(time, data[:, i])
                plt.xlabel("time(s)")
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
path = "/home/zxh/ws/quaddif/outputs/2024-12-25/14-15-56"

checkpoint_path = os.path.join(path, "checkpoints", "exported_actor.pt2")
cfg_path = os.path.join(path, ".hydra", "config.yaml")
cfg = OmegaConf.load(cfg_path)
min_action = torch.tensor([[-max_acc, -max_acc, 0]])
max_action = torch.tensor([[max_acc, max_acc, 40]])

@torch.no_grad()
def main_export():
    rospy.init_node('accel_ctrl', anonymous=True)
    node = AccelControlNode(freq=control_freq)
    logger = Logger()
    rate = rospy.Rate(control_freq)

    rospy.loginfo("Loading actor...")
    actor = torch.jit.load(checkpoint_path, map_location=torch.device('cpu'))
    rospy.loginfo("Actor loaded, warming up...")
    actor.eval()
    fake_input = torch.rand(10)
    for _ in range(10):
        actor(fake_input)
    actor.reset()
    rospy.loginfo("Actor warmed up.")
    
    vel_ema = torch.zeros(1, 3)
    while not rospy.is_shutdown():
        target_vel, quat_xyzw, vel, acc = node.get_state()
        vel_ema.lerp_(vel.unsqueeze(0), 0.1)
        state = torch.cat([target_vel, quat_xyzw, vel], dim=-1)
        raw_action: torch.Tensor = actor(state)
        action = actor.rescale(raw_action, min_action, max_action)
        node.acc_cmd = Point(x=action[0, 0].item(), y=action[0, 1].item(), z=action[0, 2].item())
        # orientation = F.normalize(vel_ema, dim=-1)
        orientation = F.normalize(vel_ema, dim=-1).lerp(torch.tensor([[1., 0., 0.]]), target_vel.norm(dim=-1).neg().exp().item()) # soft yaw switch
        # orientation = vel_ema if target_vel.norm(dim=-1).item() > 1.5 else torch.tensor([[1., 0., 0.]]) # hard yaw switch
        quat_xyzw_cmd, acc_norm = actor.post_process(action, orientation)
        thrust = acc_norm.mul(thrust_factor).clamp(0, 1).item()
        x, y, z, w = list(map(lambda x: x.item(), quat_xyzw_cmd[0].unbind(dim=-1)))
        terminated = node.command_pose(Quaternion(x=x, y=y, z=z, w=w), thrust)
        logger.log(node)
        if node.offboard_setpoint_counter % (node.freq//5) == 0:
            print(f"|time= {node.offboard_setpoint_counter / node.freq:6.2f}", end=" |")
            print("pos= " + " ".join(map(lambda x: f"{x:5.2f}", [node.pos.x, node.pos.y, node.pos.z])), end=" |")
            print("vel= " + " ".join(map(lambda x: f"{x:5.2f}", [node.vel.x, node.vel.y, node.vel.z])), end=" |")
            print("acc= " + " ".join(map(lambda x: f"{x:5.2f}", [node.acc.x, node.acc.y, node.acc.z])), end=" |")
            print("action= " + " ".join(map(lambda x: f"{x.item():5.2f}", action.unbind(dim=-1))), end=" |")
            print("thrust= " + f"{thrust:.2f} |")
        if terminated:
            logger.save_and_plot("./outputs/")
            break
        rate.sleep()

if __name__ == "__main__":
    main_export()