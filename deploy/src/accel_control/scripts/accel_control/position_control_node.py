import time
from typing import Tuple

import numpy as np
import torch
import onnxruntime as ort

import rospy
from geometry_msgs.msg import Point, Quaternion

from flight_control.FlightControl import FlightControlNode

def lerp(start: np.ndarray, end: np.ndarray, weight: float) -> np.ndarray:
    return start + weight * (end - start)

def normalize(v: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    norm = np.linalg.norm(v, axis=axis, keepdims=keepdims)
    return v / (norm + 1e-6)

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
        self.vel_ema_ratio = 0.2
        self.thrust_cmd = 0
        self.terminated = False
        self.thrust_factor = hover_thrust / 9.81
        self.device = device
        self.dtype = torch.float32
        self.factory_kwargs = {"device": device, "dtype": self.dtype}
        self.need_hidden: bool
        self.actor_loaded = False
        
        self.min_action = np.array([[-max_acc, -max_acc, 0]])
        self.max_action = np.array([[max_acc, max_acc, 40]])
        self.vel_ema = np.zeros((1, 3))
    
    @property
    def t(self):
        return self.offboard_setpoint_counter / self.freq
        
    def load_actor(self):
        if self.model_path.endswith(".onnx"):
            self.load_actor_onnx()
        elif self.model_path.endswith(".pt") or self.model_path.endswith(".pt2"):
            self.load_actor_jit()
        else:
            raise ValueError("Unsupported model format. Please use .onnx or .pt/.pt2.")
        self.actor_loaded = True
    
    def inference(self, state: np.ndarray, orientation: np.ndarray):
        if self.model_path.endswith(".onnx"):
            return self.inference_onnx(state, orientation)
        elif self.model_path.endswith(".pt") or self.model_path.endswith(".pt2"):
            return self.inference_jit(state, orientation)
    
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
    
    def inference_onnx(self, state: np.ndarray, orientation: np.ndarray):
        named_inputs = {
            "state": state.astype(np.float32),
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
            torch.rand(1, 3, **self.factory_kwargs),
            torch.rand(1, 3, **self.factory_kwargs),
            torch.rand(1, 3, **self.factory_kwargs)]
        if self.need_hidden:
            warmup_input.append(torch.rand(self.actor.hidden_shape, **self.factory_kwargs))
            self.hidden = torch.zeros(self.actor.hidden_shape, **self.factory_kwargs)
        for _ in range(10):
            self.actor(*warmup_input)
        rospy.loginfo("Actor warmed up.")
    
    def inference_jit(self, state: np.ndarray, orientation: np.ndarray):
        state = torch.from_numpy(state).to(**self.factory_kwargs)
        orientation = torch.from_numpy(orientation).to(**self.factory_kwargs)
        min_action = torch.from_numpy(self.min_action).to(**self.factory_kwargs)
        max_action = torch.from_numpy(self.max_action).to(**self.factory_kwargs)
        with torch.no_grad():
            if self.need_hidden:
                action, quat_xyzw_cmd, acc_norm, hidden = self.actor(state, orientation, min_action, max_action, self.hidden)
                self.hidden = hidden
            else:
                action, quat_xyzw_cmd, acc_norm = self.actor(state, orientation, min_action, max_action)
        action = action.squeeze(0).cpu().numpy()
        quat_xyzw_cmd = quat_xyzw_cmd.squeeze(0).cpu().numpy()
        acc_norm = acc_norm.squeeze(0).cpu().numpy()
        return action, quat_xyzw_cmd, acc_norm
    
    def step(self):
        tic = time.time()
        target_vel, quat_xyzw, vel = self.get_state()
        self.vel_ema = lerp(self.vel_ema, vel.reshape(1, 3), 0.2)
        state = np.concatenate([target_vel, quat_xyzw, vel], axis=-1).reshape(1, -1)
        target_direction = normalize(target_vel, axis=-1)
        forward = normalize(self.vel_ema, axis=-1)
        zero_yaw = np.array([[1., 0., 0.]])
        # XXX
        # orientation = forward
        # orientation = lerp(forward, target_direction, 0.8)
        orientation = lerp(forward, zero_yaw, np.exp(-np.linalg.norm(target_vel/5)))
        # orientation = self.vel_ema if np.linalg.norm(target_vel) > 1.5 else zero_yaw # hard yaw switch
        
        action, quat_xyzw_cmd, acc_norm = self.inference(state, orientation)
        
        self.acc_cmd = Point(x=action[0], y=action[1], z=action[2])
        thrust = min(1, max(0, acc_norm * self.thrust_factor))
        x, y, z, w = quat_xyzw_cmd
        terminated = self.command_pose(Quaternion(x=x, y=y, z=z, w=w), thrust)
        return time.time() - tic, terminated
    
    def acc_ctrl_cb(self, event: rospy.timer.TimerEvent):
        if self.actor_loaded:
            self.offboard_setpoint_counter += 1
        # self.update_target()
        
        if self.offboard_setpoint_counter == 5*self.freq:
            if not (self.set_mode("OFFBOARD") and self.arm()):
                rospy.logfatal("Failed to switch to OFFBOARD mode and arm the vehicle.")
                exit(1)
                
        if self.t < 20:
            self.set_pos(self.home)
        elif self.t < 60:
            self.set_pose(quat=self.pose_cmd, thrust=self.thrust_cmd)
        elif self.t < 70:
            self.set_pos(self.home)
        else:
            self.terminated = True
    
    def update_target(self):
        H = 10
        
        # R = 2
        # self.target = Point(
        #     x=R * math.sin(1.4 * t),
        #     y=R * math.cos(1.4 * t),
        #     z=H)
        t = int(self.t)
        L = 7
        if (t - 20) % 20 <= 5:
            self.target = Point(-L, L, H)
        elif (t - 20) % 20 <= 10:
            self.target = Point(L, L, H)
        elif (t - 20) % 20 <= 15:
            self.target = Point(L, -L, H)
        elif (t - 20) % 20 <= 20:
            self.target = Point(-L, -L, H)
    
    def command_pose(self, quat: Quaternion, thrust: float):
        self.pose_cmd = quat
        self.thrust_cmd = thrust
        return self.terminated
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_relpos = np.array([
            self.target.x - self.pos.x,
            self.target.y - self.pos.y,
            self.target.z - self.pos.z])
        target_vel = target_relpos / max(np.linalg.norm(target_relpos) / self.max_vel, 1)
        quat_xyzw = np.array([
            self.quat_xyzw.x,
            self.quat_xyzw.y,
            self.quat_xyzw.z,
            self.quat_xyzw.w])
        vel = np.array([
            self.vel.x,
            self.vel.y,
            self.vel.z])
        return target_vel, quat_xyzw, vel