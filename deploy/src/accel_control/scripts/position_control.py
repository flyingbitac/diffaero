#! /home/zxh/miniconda3/envs/ros1/bin/python 

import os
import datetime

import torch
from line_profiler import LineProfiler

import rospy

from accel_control.logger import Logger
from accel_control.position_control_node import PositionControlNode

def main():
    rospy.init_node('position_control_node', anonymous=True)
    control_freq = rospy.get_param("~control_freq") # Hz
    use_cuda = rospy.get_param("~use_cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    checkpoint_path = rospy.get_param("~path") # XXX
    
    node = PositionControlNode(
        freq=control_freq,
        model_path=checkpoint_path,
        home_x=rospy.get_param("~home_x"),
        home_y=rospy.get_param("~home_y"),
        target_x=rospy.get_param("~target_x"),
        target_y=rospy.get_param("~target_y"),
        height=rospy.get_param("~height"),
        max_acc=rospy.get_param("~max_acc"),
        max_vel=rospy.get_param("~max_vel"),
        hover_thrust=rospy.get_param("~hover_thrust"),
        device=device)
    node.load_actor()
    
    profiler = LineProfiler()
    profiler.add_function(node.inference_jit)
    profiler.add_function(node.inference_onnx)
    profiler.add_function(node.get_state)
    step = profiler(node.step)
    logger = Logger()
    rate = rospy.Rate(control_freq)
    
    while not rospy.is_shutdown():
        inference_time, terminated = step()
        logger.log(node)
        if node.offboard_setpoint_counter % (node.freq//5) == 0:
            print(f"|time= {node.offboard_setpoint_counter / node.freq:6.2f}", end=" |")
            print("pos= " + " ".join(map(lambda x: f"{x:5.2f}", [node.pos.x, node.pos.y, node.pos.z])), end=" |")
            print("vel= " + " ".join(map(lambda x: f"{x:5.2f}", [node.vel.x, node.vel.y, node.vel.z])), end=" |")
            print("action= " + " ".join(map(lambda x: f"{x:5.2f}", [node.acc_cmd.x, node.acc_cmd.y, node.acc_cmd.z])), end=" |")
            print(f"thrust= {node.thrust_cmd:.2f} |inference_time= {inference_time*1000:6.2f}.ms |")
            if inference_time > 1/control_freq:
                rospy.logwarn("Inference time exceeds control period.")
        if terminated:
            now = datetime.datetime.now()
            path = os.path.join("./outputs/", now.strftime('%Y-%m-%d'), now.strftime('%H-%M-%S'))
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "runtime_profile.txt"), "w", encoding="utf-8") as f:
                profiler.print_stats(stream=f, output_unit=1e-3)
            logger.save_and_plot(path)
            break
        rate.sleep()

if __name__ == "__main__":
    main()