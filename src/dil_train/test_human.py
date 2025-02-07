#!/usr/bin/env python
import rospy
import time
import pandas as pd
import csv
from std_msgs.msg import Float64MultiArray
mode = 'walk'
# Those experiments are run for several purposes. 
# 1 - for Table (50 experiments), choose video = 'no'
# 2 - for Video (10 experiments), choose video = 'simple'
# 3 - for Video (extra video to show the human coming real close to the robot), choose video = 'extra'

video = 'no'
if mode=='sit':
    pos = []
    print("Experiment with sitting human")
    # Load test data:
    pos_181 = '/home/robot/Documents/Spheres_Bones_rotated_0.csv'
    pos_182 = '/home/robot/Documents/Spheres_Bones_rotated_1.csv'
    pos_183 = '/home/robot/Documents/Spheres_Bones_rotated_2.csv'

    # TEST
    pos_181 = pd.read_csv(pos_181, quoting=csv.QUOTE_NONNUMERIC)
    pos_181 = pos_181.to_numpy()
    pos.append(pos_181)
    a = 1000
    rep = 50
    dt = 120 
else:
    pos = []
    print("Experiment with walking human")
    if video=='simple':
        pos_183 = '/home/robot/workspaces/human_data/Participant_8410_csv/Participant_8410_Setup_A_Seq_1_Trial_3.xsens.bvh.csv'
        pos_183 = pd.read_csv(pos_183, quoting=csv.QUOTE_NONNUMERIC)
        pos_183 = pos_183.to_numpy()
        pos.append(pos_183)
        a = 1000
        rep = 10
        dt = 125
        start = 12000
        d_y = 1.0
    if video=='extra':
        pos_183 = '/home/robot/workspaces/human_data/Participant_8410_csv/Participant_8410_Setup_A_Seq_1_Trial_3.xsens.bvh.csv'
        pos_183 = pd.read_csv(pos_183, quoting=csv.QUOTE_NONNUMERIC)
        pos_183 = pos_183.to_numpy()
        pos.append(pos_183)
        a = 1000
        rep = 5
        dt = 125
        start = 12000
        d_y = 0.8
    else:
        pos_181 = '/home/robot/workspaces/human_data/Participant_8410_csv/Participant_8410_Setup_A_Seq_1_Trial_1.xsens.bvh.csv'
        pos_182 = '/home/robot/workspaces/human_data/Participant_8410_csv/Participant_8410_Setup_A_Seq_1_Trial_2.xsens.bvh.csv'
        pos_183 = '/home/robot/workspaces/human_data/Participant_8410_csv/Participant_8410_Setup_A_Seq_1_Trial_3.xsens.bvh.csv'
        pos_184 = '/home/robot/workspaces/human_data/Participant_8410_csv/Participant_8410_Setup_A_Seq_1_Trial_4.xsens.bvh.csv'
        pos_185 = '/home/robot/workspaces/human_data/Participant_8410_csv/Participant_8410_Setup_A_Seq_1_Trial_5.xsens.bvh.csv'
        pos_186 = '/home/robot/workspaces/human_data/Participant_8410_csv/Participant_8410_Setup_A_Seq_2_Trial_1.xsens.bvh.csv'
        pos_187 = '/home/robot/workspaces/human_data/Participant_8410_csv/Participant_8410_Setup_A_Seq_2_Trial_2.xsens.bvh.csv'
        pos_188 = '/home/robot/workspaces/human_data/Participant_8410_csv/Participant_8410_Setup_A_Seq_2_Trial_3.xsens.bvh.csv'
        pos_189 = '/home/robot/workspaces/human_data/Participant_8410_csv/Participant_8410_Setup_A_Seq_2_Trial_4.xsens.bvh.csv'
        pos_190 = '/home/robot/workspaces/human_data/Participant_8410_csv/Participant_8410_Setup_A_Seq_2_Trial_5.xsens.bvh.csv'
        pos_181 = pd.read_csv(pos_181, quoting=csv.QUOTE_NONNUMERIC)
        pos_181 = pos_181.to_numpy()
        pos.append(pos_181)
        pos_182 = pd.read_csv(pos_182, quoting=csv.QUOTE_NONNUMERIC)
        pos_182 = pos_182.to_numpy()
        pos.append(pos_182)
        pos_183 = pd.read_csv(pos_183, quoting=csv.QUOTE_NONNUMERIC)
        pos_183 = pos_183.to_numpy()
        pos.append(pos_183)
        pos_184 = pd.read_csv(pos_184, quoting=csv.QUOTE_NONNUMERIC)
        pos_184 = pos_184.to_numpy()
        pos.append(pos_184)
        pos_185 = pd.read_csv(pos_185, quoting=csv.QUOTE_NONNUMERIC)
        pos_185 = pos_185.to_numpy()
        pos.append(pos_185)
        pos_186 = pd.read_csv(pos_186, quoting=csv.QUOTE_NONNUMERIC)
        pos_186 = pos_186.to_numpy()
        pos.append(pos_186)
        pos_187 = pd.read_csv(pos_187, quoting=csv.QUOTE_NONNUMERIC)
        pos_187 = pos_187.to_numpy()
        pos.append(pos_187)
        pos_188 = pd.read_csv(pos_188, quoting=csv.QUOTE_NONNUMERIC)
        pos_188 = pos_188.to_numpy()
        pos.append(pos_188)
        pos_189 = pd.read_csv(pos_189, quoting=csv.QUOTE_NONNUMERIC)
        pos_189 = pos_189.to_numpy()
        pos.append(pos_189)
        pos_190 = pd.read_csv(pos_190, quoting=csv.QUOTE_NONNUMERIC)
        pos_190 = pos_190.to_numpy()
        pos.append(pos_190)
        d_y = 0.7
        a = 3000
        rep = 5
        dt = 125
        start = 0

    # split data to the equal parts:
    splited_data=[]
    for k in range (len(pos)):
        temp = pos[k]
        for i in range(rep):
            splited_data.append(temp[start+i*a:i*a+start+10000])

print(len(splited_data))

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

    def step(self,i,data_part,file_number,mode):
        point_array = [0]*43
        for a in range(14):
            if mode=='sit':
                point_array[3*a] = (data_part[i][3*a])+0.5
                point_array[3*a+1] = (data_part[i][3*a+1])+0.5
                point_array[3*a+2] = (data_part[i][3*a+2])
            else:
                point_array[3*a] = (data_part[i][3*a])+1.0
                point_array[3*a+1] = (data_part[i][3*a+1])+d_y
                point_array[3*a+2] = (data_part[i][3*a+2])-1.2
        point_array[42] = file_number+1
        obstacle_data = Float64MultiArray()
        obstacle_data.data = point_array
        self.pub.publish(obstacle_data)

if __name__ == '__main__':
    rospy.init_node("human_poses_provider", anonymous=True)
    env = ENV()
    i = 0
    cond_temp=0
    rate = rospy.Rate(dt) #hz
    msg = rospy.wait_for_message("/flag", Float64MultiArray)
    if(msg):
        while not rospy.is_shutdown():
            condition_h = env.check_condition()
            if condition_h[1]==1:
                time.sleep(0.02)
                i=0
            else:
                temp = splited_data[condition_h[0]]
                env.step(i,temp,condition_h[0],mode)
                print(condition_h,i)
                i+=1
            rate.sleep()


