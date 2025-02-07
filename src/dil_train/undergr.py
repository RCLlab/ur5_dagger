#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import stream_tee as stream_tee
import __main__ as main
import rospy
from std_msgs.msg import Float64MultiArray, String
import time
torch.manual_seed(1)
from stream_tee import write_mat
from initialization import set_init_pose
from cascadi_solver import cascadi_solv

class MyModel(nn.Module):
    def __init__(self, dev, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.dev = dev
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        current_dim = input_dim
        self.layers = nn.ModuleList()
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim,hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim,output_dim))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        out = self.layers[-1](x)
        return out
        
class ENV:
    def __init__(self,run_name):
        rospy.Subscriber('/info', Float64MultiArray, self.callback)
        self.pub = rospy.Publisher('/ur_driver/URScript', String, queue_size=1)
        self.flag_pub = rospy.Publisher('/flag', Float64MultiArray, queue_size=1)
        self.run_name = run_name
        self.A = [0.0, -2.3, -1.1, -1.2, -1.2, 0.5]
        self.B = [3.0, -1.6, -1.7, -1.7, -1.7, 1.0]
        self.init_poses=[0,-1.5708,0,-1.5708,0,0]
        self.net = [48,36,27]
        self.start = self.B
        self.goal = self.A
        self.i = 0
        self.first = 0
        self.init_log_variables()
        self.total = 0
        self.threshold_1 = 0.18
        self.threshold_2 = 0.12
        self.hello_str=[-1,0]
        self.t_total = time.time()

    def callback(self, data):
        self.observation = data.data[0:55]

    def choose_model(self, model_type):
        self.save_dir = model_type
        print(model_type)

        if model_type == 'DNN_P/':
            self.model = MyModel(dev,48,6,[48,36,27]).to(dev)
            self.AB_model = '/home/robot/workspaces/Big_Data/nn_train/log/DNN_P/20220914_170026/model.pth'
            self.BA_model = '/home/robot/workspaces/Big_Data/nn_train/log/DNN_P/20220914_205108/model.pth'
            self.to_A_model = '/home/robot/workspaces/Big_Data/nn_train/log/DNN_0/20220914_214713/model.pth'
            self.to_B_model = '/home/robot/workspaces/Big_Data/nn_train/log/DNN_0/20220914_221451/model.pth'
        
        self.model.cuda()
        if self.start==self.A:
            self.model.load_state_dict(torch.load(self.AB_model))
        else:
            self.model.load_state_dict(torch.load(self.BA_model))

    def done(self):
        max_diff = 0
        temp = 0
        arrive = False
        for i in range(6):
            temp = abs(self.goal[i]-self.observation[i])
            if temp>max_diff:
                max_diff = temp
        print(max_diff, self.start[0],self.A[0])
        if max_diff<self.threshold_1 and max_diff>self.threshold_2:
            if self.start[0]==self.A[0]:
                self.point_dir = 'AB/'
                self.model.load_state_dict(torch.load(self.to_B_model))
            else:
                self.point_dir = 'BA/'
                self.model.load_state_dict(torch.load(self.to_A_model))
        if max_diff<self.threshold_2:
            # print("-----Arrived------")
            arrive = True
            if self.start[0]==self.A[0]:
                self.model.load_state_dict(torch.load(self.AB_model))
            else:
                self.model.load_state_dict(torch.load(self.BA_model))
        return arrive

    def test(self, x_test):
        self.model.eval()
        prediction=[]
        with torch.no_grad():
            seq_data = np.array(x_test)
            seq_data = torch.tensor([i for i in seq_data], dtype=torch.float32).to(dev)
            prediction = self.model(seq_data)
        return prediction

    def step(self):
        t_nn = time.time()
        u = self.test(self.observation[0:48])
        elapsed_nn = time.time() - t_nn
        temp = list(u.cpu().numpy())
        vel = [temp[0],temp[1],temp[2],temp[3],temp[4],temp[5]]
        # vel, min_dist, lin_vel, T = cascadi_solv(t2,self.observation[0:48])
        hello_str = "speedj(["+str(vel[0])+","+str(vel[1])+","+str(vel[2])+","+str(vel[3])+","+str(vel[4])+","+str(vel[5])+"],"+"5.0"+",0.1)" 
        elapsed_time = time.time() - self.t_total
        self.pub.publish(hello_str)
        # temp2 = [vel[0],vel[1],vel[2],vel[3],vel[4],vel[5]]
        self.nn_actions.append(temp)
        # self.casadi.append(temp2)
        # self.minimum_dist.append(min_dist)
        # self.lin_vels.append(lin_vel)
        self.joint_poses.append(self.observation[0:6])
        self.real_vels.append(self.observation[48:54])
        self.human_poses.append(self.observation[6:48])
        self.nn_time.append(elapsed_nn)
        self.goals.append(self.goal)
        if self.goal[1]==self.A[1]:
            self.init_poses=self.B
        else:
            self.init_poses=self.A
        self.file_n.append(self.observation[54])
        self.time.append(elapsed_time)

    def reset(self):   
        print("reset")
        if self.first<1:
            self.first+=1
            set_init_pose(self.start[0:6],6)
            self.threshold_1 = 0.55
            self.threshold_2 = 0.12
            time.sleep(5)
            self.init_log_variables()
            self.t_total = time.time()
        else:
            time.sleep(0.5)
            self.hello_str[1] = 1
            pub_data = Float64MultiArray()
            pub_data.data = self.hello_str
            self.flag_pub.publish(pub_data)
            self.save_log(self.i)
            set_init_pose(self.start[0:6], 6)
            time.sleep(0.5)
            self.init_log_variables()
            self.threshold_1 = 0.18
            self.threshold_2 = 0.02
            self.hello_str[0]+= 1
            self.hello_str[1] = 0
            pub_data.data = self.hello_str
            self.flag_pub.publish(pub_data)
            time.sleep(1)
            self.i+=1
        self.step()
    
    def init_log_variables(self):
        self.nn_actions = []
        self.joint_poses = []
        self.human_poses = []
        self.real_vels = []
        self.nn_time = []
        self.observation = [1]*55
        self.minimum_dist = []
        self.lin_vels = []
        self.tp = [0 for c in range(21)]
        self.goals = []
        self.arrive = False
        self.file_n=[]
        self.time = []
        self.casadi = []

    def save_log(self,save_iter):
        rec_dir = '/home/robot/workspaces/Big_Data/nn_train/'
        os.chdir(rec_dir)
        print("***saving***")

        write_mat('test_log/' + self.save_dir + self.point_dir + self.run_name,
                        {'actions': self.nn_actions,
                        'joint_positions': self.joint_poses,
                        'human_poses':self.human_poses,
                        'real_vels': self.real_vels,
                        'goal':self.goals,
                        'nn_time':self.nn_time,
                        'network':self.net,
                        'file_n':self.file_n,
                        'min_dist':self.minimum_dist,
                        'lin vels':self.lin_vels,
                        'casadi':self.casadi,
                        'time':self.time},
                        str(save_iter))    

if __name__ == '__main__':
    rospy.init_node("pytorch_test", anonymous=True)
    run_name = stream_tee.generate_timestamp()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = ENV(run_name)
    model_type = 'DNN_P/'
    env.choose_model(model_type)
    t = time.time()
    env.reset()
    i = 0
    save_iter = 0
    rate = rospy.Rate(20) #hz
    while not rospy.is_shutdown():
        done = env.done()
        if done==True:
            elapsed = time.time() - t
            print("Episode ", i, ' time = ', elapsed)
            i+=1
            print("Episode", i, " is started")
            env.reset()
            t = time.time()
        else:
            env.step()
        
        rate.sleep()

        