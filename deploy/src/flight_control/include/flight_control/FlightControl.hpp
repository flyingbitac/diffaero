#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Point.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <string.h>
#include <vector>
#include <deque>
#include <boost/bind.hpp>

// 存储空间三坐标位置的数据结构
struct Point{
    double x, y, z;
    Point(): x(0), y(0), z(0) {}
    Point(double x, double y, double z): x(x), y(y), z(z) {}
    Point(const Point& p){
        x = p.x;
        y = p.y;
        z = p.z;
    }
    Point operator+(const Point& p){
        Point sum(x+p.x, y+p.y, z+p.z);
        return sum;
    }
    Point operator-(const Point& p){
        Point sum(x-p.x, y-p.y, z-p.z);
        return sum;
    }
    Point operator/(const double d){
        Point quotient(x/d, y/d, z/d);
        return quotient;
    }
    Point operator*(const double d){
        Point prod(x*d, y*d, z*d);
        return prod;
    }
    Point operator*(const Point &p){
        Point prod(x*p.x, y*p.y, z*p.z);
        return prod;
    }
    Point& operator=(const Point& p){
        x = p.x;
        y = p.y;
        z = p.z;
        return *this;
    }
    void operator-=(const Point& p){
        x -= p.x;
        y -= p.y;
        z -= p.z;
    }
    void operator+=(const Point& p){
        x += p.x;
        y += p.y;
        z += p.z;
    }
};

Point operator*(const double d, const Point& p){
    Point prod(p.x*d, p.y*d, p.z*d);
    return prod;
}

Point power(const Point &p, int a){
    Point pow = p;
    for (int i = 0; i < a-1; i++)
        pow = pow * p;
    return pow;
}

double distance(const Point &a, const Point &b){
    double dist = 0;
    dist += pow(a.x - b.x, 2);
    dist += pow(a.y - b.y, 2);
    dist += pow(a.z - b.z, 2);
    return sqrt(dist);
}

typedef std::vector<double> vd;
typedef std::vector<Point> vp;

class FlightControlNode{
public:
    Point pos, vel;

    ros::NodeHandle nh;
    ros::Subscriber state_sub;
    ros::Subscriber odom_sub;
    ros::Publisher pos_pub;
    ros::Publisher vel_pub;
    ros::Publisher acc_pub;
    ros::ServiceClient arming_client;
    ros::ServiceClient set_mode_client;
    mavros_msgs::State current_state;
    mavros_msgs::CommandBool arm_cmd;
    mavros_msgs::SetMode offb_set_mode;

    FlightControlNode();
    static void state_cb(const mavros_msgs::State::ConstPtr &, FlightControlNode*);
    static void odom_cb(const nav_msgs::Odometry::ConstPtr &, FlightControlNode*);
    void set_pos(double px, double py, double pz);
    void set_pos(const Point& pos);
    void set_vel(double vx, double vy, double vz);
    void set_vel(const Point& vel);
    void set_acc(double ax, double ay, double az);
    void set_acc(const Point& acc);
    bool arm();
    bool disarm();
    bool set_mode(std::string mode);
};