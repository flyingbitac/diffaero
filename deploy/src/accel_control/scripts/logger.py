
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Logger:
    def __init__(self):
        self.ts = []
        self.pos = []
        self.vel = []
        self.acc = []
        self.acc_cmd = []
        self.thrusts = []
        self.euler = []
    
    def log(self, node):
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
        euler = np.array(self.euler) * 180 / np.pi
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
                if name == "euler angles":
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