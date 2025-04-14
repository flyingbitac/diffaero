import rospy
from geometry_msgs.msg import Point

from flight_control.FlightControl import FlightControlNode

def main():
    rospy.init_node('offb_node', anonymous=True)
    node = FlightControlNode()
    
    rate = rospy.Rate(20)
    while not rospy.is_shutdown() and not node.current_state.connected:
        rate.sleep()
    
    for _ in range(100):
        node.set_pos(Point(0.0, 0.0, 2.0))
        rate.sleep()
    
    last_request = rospy.Time.now()
    
    while not rospy.is_shutdown():
        if node.current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_request > rospy.Duration(5.0)):
            node.set_mode()
            last_request = rospy.Time.now()
        else:
            if not node.current_state.armed and (rospy.Time.now() - last_request > rospy.Duration(5.0)):
                node.arm()
                last_request = rospy.Time.now()
        
        node.set_pos(Point(0.0, 0.0, 4.0))
        print(node.pos.z)
        
        rate.sleep()

if __name__ == '__main__':
    main()