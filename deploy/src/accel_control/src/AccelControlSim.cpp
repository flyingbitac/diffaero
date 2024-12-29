#include <accel_control/AccelControl.hpp>

using namespace std;

class AccelControlNode: public FlightControlNode{
public:
    int offboard_setpoint_counter; // 为系统提供初始化时间
    ros::Timer pos_ctrl_timer; // 位置控制

    const Point home; // 起始点

    AccelControlNode(): home(0, 0, 5), offboard_setpoint_counter(0){
        pos_ctrl_timer = nh.createTimer(ros::Duration(0.1), bind(&AccelControlNode::pos_ctrl_cb, this), false);
    }
    static void pos_ctrl_cb(AccelControlNode*);
};

// 位置控制回调函数，每0.1秒发出一个未知指令
// 把无人机指引到home或destination的位置
void AccelControlNode::pos_ctrl_cb(AccelControlNode* node){
    node->offboard_setpoint_counter++;
    if (node->offboard_setpoint_counter == 10){
        if (!node->set_mode("OFFBOARD") || !node->arm()){
            ROS_FATAL("Failed to switch to OFFBOARD mode and arm the vehicle.");
            exit(EXIT_FAILURE);
        }
    }
    if (node->offboard_setpoint_counter <= 200){
        node->set_pos(node->home);
    }else{
        double radius = 3;
        double theta = double(node->offboard_setpoint_counter) / 30.;
        node->set_acc(radius * sin(theta), radius * cos(theta), cos(theta * 3));
    }
}

int main(int argc, char **argv){
    ros::init(argc, argv, "accel_control");
    AccelControlNode node;
    ros::spin();
    return 0;
}