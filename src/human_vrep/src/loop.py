#!/usr/bin/env python
import rospy
import time
import pandas as pd
import csv
from std_msgs.msg import Float64MultiArray

datafile = '/home/robot/ur5_dagger/closefar.csv'
pos = pd.read_csv(datafile, quoting=csv.QUOTE_NONNUMERIC)
pos = pos.to_numpy()
p_len = len(pos)
print(len(pos))

print("start the high level controller!")

class ENV:
    def __init__(self):
        rospy.Subscriber('/flag', Float64MultiArray, self.callback)
        self.pub = rospy.Publisher('/Obstacle/human_spheres', Float64MultiArray, queue_size=1)
        self.iter = 0
        self.condition_h = [0]*2

    def callback(self, data):
        self.condition_h = data.data[0:2]

    def check_condition(self):
        return [int(self.condition_h[0]),self.condition_h[1]]

    def step(self,i,datas):
        point_array = [0]*465
        for a in range(14):
            point_array[3*a] = (datas[i][3*a+1])-0.4
            point_array[3*a+1] = (datas[i][3*a])+2.0
            point_array[3*a+2] = (datas[i][3*a+2])
        for f in range(10):
            for a in range(14): 
                point_array[f*42+43+3*a] = (datas[i][3*a+1])-0.4
                point_array[f*42+43+3*a+1] = (datas[i][3*a])+2.0
                point_array[f*42+43+3*a+2] = (datas[i][3*a+2])
        point_array[463] = 0
        point_array[464] = 0
        obstacle_data = Float64MultiArray()
        obstacle_data.data = point_array
        self.pub.publish(obstacle_data)

if __name__ == '__main__':
    rospy.init_node("human_poses_provider", anonymous=True)
    env = ENV()
    i = 0
    cond_temp=0
    rate = rospy.Rate(125) #hz
    msg = rospy.wait_for_message("/flag", Float64MultiArray)
    if(msg):
        while not rospy.is_shutdown():
            condition_h = env.check_condition()
            if i>p_len-1:
                i=0
            if condition_h[1]==1:
                time.sleep(0.02)
            else:
                env.step(i,pos)
                print(condition_h,i)
                i+=1
            rate.sleep()
