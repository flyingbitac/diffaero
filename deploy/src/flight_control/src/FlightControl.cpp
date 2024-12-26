#include <flight_control/FlightControl.hpp>

FlightControlNode::FlightControlNode(){
    this->state_sub = nh.subscribe<mavros_msgs::State>
        ("mavros/state", 10, bind(&state_cb, _1, this));
    this->odom_sub = nh.subscribe<nav_msgs::Odometry>
        ("mavros/odometry/in", 10, bind(&odom_cb, _1, this));
    this->pos_pub = nh.advertise<geometry_msgs::PoseStamped>
        ("mavros/setpoint_position/local", 10);
    this->vel_pub = nh.advertise<geometry_msgs::TwistStamped>
        ("mavros/setpoint_velocity/cmd_vel", 10);
    this->acc_pub = nh.advertise<mavros_msgs::PositionTarget>
        ("mavros/setpoint_raw/local", 10);
    this->arming_client = nh.serviceClient<mavros_msgs::CommandBool>
        ("mavros/cmd/arming");
    this->set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
        ("mavros/set_mode");
}

// 里程计回调函数，更新当前无人机的位置和线速度
void FlightControlNode::odom_cb(
    const nav_msgs::Odometry::ConstPtr &msg,
    FlightControlNode *node
){
    node->pos.x = msg->pose.pose.position.x;
    node->pos.y = msg->pose.pose.position.y;
    node->pos.z = msg->pose.pose.position.z;
    node->vel.x = msg->twist.twist.linear.x;
    node->vel.y = msg->twist.twist.linear.y;
    node->vel.z = msg->twist.twist.linear.z;
    return;
}

// 状态回调函数，记录无人机控制状态等信息
void FlightControlNode::state_cb(
    const mavros_msgs::State::ConstPtr& msg,
    FlightControlNode *node
){
    node->current_state = *msg;
}

// 定点飞行函数
void FlightControlNode::set_pos(double px, double py, double pz){
    geometry_msgs::PoseStamped msg;
    msg.pose.position.x = px;
    msg.pose.position.y = py;
    msg.pose.position.z = pz;
    msg.header.stamp = ros::Time::now();
    pos_pub.publish(msg);
    return;
}

// 定点飞行函数
void FlightControlNode::set_pos(const Point& pos){
    this->set_pos(pos.x, pos.y, pos.z);
}

// 定速飞行函数
void FlightControlNode::set_vel(double vx, double vy, double vz){
    geometry_msgs::TwistStamped msg;
    msg.twist.linear.x = vx;
    msg.twist.linear.y = vy;
    msg.twist.linear.z = vz;
    msg.header.stamp = ros::Time::now();
    vel_pub.publish(msg);
    return;
}

// 定速飞行函数
void FlightControlNode::set_vel(const Point& vel){
    this->set_vel(vel.x, vel.y, vel.z);
}

// 姿态油门控制函数
void FlightControlNode::set_acc(double ax, double ay, double az){
    mavros_msgs::PositionTarget msg;
    msg.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
    msg.type_mask = mavros_msgs::PositionTarget::IGNORE_PX |
                    mavros_msgs::PositionTarget::IGNORE_PY |
                    mavros_msgs::PositionTarget::IGNORE_PZ |
                    mavros_msgs::PositionTarget::IGNORE_VX |
                    mavros_msgs::PositionTarget::IGNORE_VY |
                    mavros_msgs::PositionTarget::IGNORE_VZ |
                    mavros_msgs::PositionTarget::IGNORE_YAW |
                    mavros_msgs::PositionTarget::IGNORE_YAW_RATE;
    msg.acceleration_or_force.x = ax;
    msg.acceleration_or_force.y = ay;
    msg.acceleration_or_force.z = az;
    msg.header.stamp = ros::Time::now();
    acc_pub.publish(msg);
    return;
}

void FlightControlNode::set_acc(const Point& acc){
    this->set_acc(acc.x, acc.y, acc.z);
}

bool FlightControlNode::arm(){
    arm_cmd.request.value = true;
    bool result = this->arming_client.call(arm_cmd);
    if (result)
        ROS_INFO("Vehicle Armed.");
    else
        ROS_INFO("Vehicle Arm Failed.");
    return result;
}

bool FlightControlNode::disarm(){
    arm_cmd.request.value = false;
    bool result = this->arming_client.call(arm_cmd);
    if (result)
        ROS_INFO("Vehicle Disarmed.");
    else
        ROS_INFO("Vehicle Disarm Failed.");
    return result;
}

bool FlightControlNode::set_mode(std::string mode="OFFBOARD"){
    offb_set_mode.request.custom_mode = mode;
    bool result = this->set_mode_client.call(offb_set_mode);
    if (result)
        ROS_INFO("Vihicle mode set to %s.", mode.c_str());
    else
        ROS_INFO("Failed to set vehicle mode to %s.", mode.c_str());
    return result;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "offb_node");
    FlightControlNode node;
    
    ros::Rate rate(20.0);
    while(ros::ok() && !node.current_state.connected){
        ros::spinOnce();
        rate.sleep();
    }
    //send a few setpoints before starting
    for(int i = 100; ros::ok() && i > 0; --i){
        node.set_pos(0., 0., 2.);
        ros::spinOnce();
        rate.sleep();
    }
    
    ros::Time last_request = ros::Time::now();
    while(ros::ok()){
        if( node.current_state.mode != "OFFBOARD" &&
            (ros::Time::now() - last_request > ros::Duration(5.0))){
            if(node.set_mode() && node.offb_set_mode.response.mode_sent){
                ROS_INFO("Offboard enabled");
            }
            last_request = ros::Time::now();
        } else {
            if( !node.current_state.armed &&
                (ros::Time::now() - last_request > ros::Duration(5.0))){
                if( node.arm() && node.arm_cmd.response.success){
                    ROS_INFO("Vehicle armed");
                }
                last_request = ros::Time::now();
            }
        }

        node.set_acc(Point(0., 0., 4));

        ros::spinOnce();
        rate.sleep();
    }
}