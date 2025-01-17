# SITL Simulation Setup for Position Control

## Step 1: Installation

Follow the official instructions in [PX4 website](https://docs.px4.io/main/en/ros/mavros_installation.html) to install ROS(`noetic` for Ubuntu 20.04 or `melodic` for 18.04), MAVROS and PX4.

After installed PX4, ROS and MAVROS, run the following command to pre-build the code for Gazebo SITL simulation:

```bash
source /opt/ros/noetic/setup.bash # for ROS noetic
source /opt/ros/melodic/setup.bash # for ROS melodic
cd /path/to/PX4-Autopilot
DONT_RUN=1 make px4_sitl_default gazebo-classic
```

## Step 2: Workspace Setup

Since the odometry data we use are from the optical tracking system, which can be seen as the ground truth of the data from the onboard IMU, we need to publish an additional topic `/ground_truth/state` to transfer the ground truth odometry data to the agent by modifying the .sdf file of the quadrotor model:

1. After creating a backup, open `PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models/iris/iris.sdf`.
2. Append the following code to the end of the plugin list:
   ```xml
    <plugin name="p3d_base_controller" filename="libgazebo_ros_p3d.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>50.0</updateRate>
      <bodyName>base_link</bodyName>
      <topicName>ground_truth/state</topicName>
      <gaussianNoise>0.01</gaussianNoise>
      <frameName>world</frameName>
      <xyzOffsets>0 0 0</xyzOffsets>
      <rpyOffsets>0 0 0</rpyOffsets>
    </plugin>
   ```

Then, build the code for deployment as ROS packages:
```bash
source /opt/ros/noetic/setup.bash # for ROS noetic
source /opt/ros/melodic/setup.bash # for ROS melodic
cd /path/to/quaddif/deploy
catkin build # -DPYTHON_EXECUTABLE=$(which python) if you're using anaconda venv
```

## Step 3: Start the Simulation

To start ROSCORE, MAVROS, PX4 and Gazebo, run:
```bash
source /opt/ros/noetic/setup.bash # for ROS noetic
source /opt/ros/melodic/setup.bash # for ROS melodic
cd /path/to/quaddif/deploy
source devel/setup.bash
export PX4_PATH=/path/to/PX4-Autopilot
source px4_setup.bash
roslaunch flight_control start_sim.launch # gui:=false for headless simulation
```

Check if MAVROS is publishing topic `/ground_truth/state`:
```bash
source /opt/ros/noetic/setup.bash # for ROS noetic
source /opt/ros/melodic/setup.bash # for ROS melodic
rostopic list | grep /ground_truth/state
```

## Step 4: Start the Offboard Control Node

After modified the model path in `quaddif/deploy/src/accel_control/scripts/position_control.py`, start the node by:
```bash
source /opt/ros/noetic/setup.bash # for ROS noetic
source /opt/ros/melodic/setup.bash # for ROS melodic
cd /path/to/quaddif/deploy
source devel/setup.bash
rosrun accel_control position_control.py
```

Once the simulation is done, the ros node should shutdown automatically, and the simulation result are logged in `quaddif/deploy/outputs`.