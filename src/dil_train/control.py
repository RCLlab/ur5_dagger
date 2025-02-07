#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

pub = rospy.Publisher('/info', Float64MultiArray, queue_size=1)
position = [0]*6
velocity = [0]*6
# human_data = [0.5]*43
human_data = [3.37788950970821,	-0.0112980330176171,	0.374769487753649,	3.29019001778140,	0.214561325289906,	0.129776374294684,	3.18321324533274,	0.0916132830225886,	0.287391017542728,	3.48576555895343,	0.188405847231161,	0.203575259337489,	3.12097533510650,	0.00344140406706139,	0.180159979937103,	3.48871658027040,	0.160264506177836,	0.0540905125210887,	3.05873742488027,	-0.0847304748884661,	0.0729289423314792,	3.49166760158737,	0.132123165124510,	-0.0953942342953116,	3.02323925080542,	-0.160357283516576,	-0.0707064986247592,	3.36494162929491,	0.0468059007339552,	-0.160756048744236,	2.99690486080154,	-0.273584619591562,	-0.196710733652138,	3.24914229477345,	-0.0437080289914229,	-0.253913784234094,	3.34784351883729,	0.130222350766204,	0.298094110502407,	3.24101404194146,	0.260282082317083,	-0.0310907407432772]

def callback(data):
    global position, velocity
    position = data.position[0:6]
    velocity = data.velocity[0:6]
check = 0

def h_callback(data):
    global human_data
    for i in range(42):
        human_data[i] = data.data[i]

def main():
    global position, velocity, human_data
    rospy.init_node('control', anonymous=True)
    rospy.Subscriber("/joint_states", JointState, callback)
    rospy.Subscriber("/Obstacle/human_spheres", Float64MultiArray, h_callback)
    rate = rospy.Rate(125)
    while not rospy.is_shutdown():
        point_array = [0]*54
        point_array[0:6] = position[0:6]
        point_array[6:48] = human_data[0:42]
        point_array[48:54] = velocity[0:6]
        # point_array[54] = human_data[42]
        infodata = Float64MultiArray()
        infodata.data = point_array
        rospy.loginfo(infodata)
        pub.publish(infodata)
        rate.sleep()

if __name__ == '__main__': main()
