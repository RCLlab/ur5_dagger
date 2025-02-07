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
from stream_tee import write_mat
from initialization import set_init_pose
import random
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
    def __init__(self,run_name,dev,model,n,controller, human_mode):
        rospy.Subscriber('/info', Float64MultiArray, self.callback)
        self.pub = rospy.Publisher('/ur_driver/URScript', String, queue_size=1)
        self.flag_pub = rospy.Publisher('/flag', Float64MultiArray, queue_size=1)
        self.run_name = run_name
        self.A = [0.0, -2.3, -1.1, -1.2, -1.2, 0.5]
        self.B = [3.0, -1.6, -1.7, -1.7, -1.7, 1.0]
        self.start = self.A
        self.goal = self.B
        self.controller = controller
        self.human_mode = human_mode
        self.i = 0
        self.first = 0
        self.hello_str=[-1,0]
        self.init_log_variables()
        self.total = 0
        self.dev = dev
        self.model = model
        self.nn_layers = n
        self.min_loss = 1
        self.threshold_1 = 0.5
        self.threshold_2 = 0.12
    
    def callback(self, data):
        self.observation = data.data[0:55]

    def choose_model(self):
        print("DA-DNN/")
        self.AB_model = '/home/robot/workspaces/Big_Data/nn_train/log/Dagger/20230214_170431/model.pth'
        self.BA_model = '/home/robot/workspaces/Big_Data/nn_train/log/Dagger/20230217_013255/model.pth'
        self.to_A_model = '/home/robot/workspaces/Big_Data/nn_train/log/DNN_0/20220914_214713/model.pth'
        self.to_B_model = '/home/robot/workspaces/Big_Data/nn_train/log/DNN_0/20220914_221451/model.pth'
       
        if self.goal==self.A[0]:
            print("AB")
            self.model.load_state_dict(torch.load(self.BA_model))
        else:
            print("BA")
            self.model.load_state_dict(torch.load(self.AB_model))

    def done(self):
        max_diff = 0
        temp = 0
        arrive = False
        for i in range(6):
            temp = abs(self.goal[i]-self.observation[i])
            if temp>max_diff:
                max_diff = temp
        # print(max_diff,self.goal[0])
        if max_diff<self.threshold_2:
            if self.goal[0]==self.A[0]:
                self.goal = self.B
                self.model.load_state_dict(torch.load(self.AB_model))
                arrive = True
            else:
                self.goal = self.A
                self.model.load_state_dict(torch.load(self.BA_model))
        else:
            if max_diff<self.threshold_1:
                if self.goal[0]==self.A[0]:
                    self.model.load_state_dict(torch.load(self.to_A_model))
                else:
                    self.model.load_state_dict(torch.load(self.to_B_model))
        return arrive
    
    def test(self, x_test):
        self.model.eval()
        with torch.no_grad():
            Data = np.array(x_test)
            Data = torch.tensor([i for i in Data], dtype=torch.float32).to(dev)
            Nov = self.model(Data)
        return Nov

    def step(self):
        t_nn = time.time()
        u = self.test(self.observation[0:48])
        elapsed_nn = time.time() - t_nn
        temp = list(u.cpu().numpy())
        t2 = [temp[0],temp[1],temp[2],temp[3],temp[4],temp[5]]
        a, clv, nlv, lin_vel_limit, T, min_dist = cascadi_solv(t2,self.observation[0:48])
        vel = [a[0],a[1],a[2],a[3],a[4],a[5]]
        # vel = t2
        hello_str = "speedj(["+str(vel[0])+","+str(vel[1])+","+str(vel[2])+","+str(vel[3])+","+str(vel[4])+","+str(vel[5])+"],"+"5.0"+",0.1)" 
        elapsed_time = time.time() - self.t_total
        self.pub.publish(hello_str)
        self.nn_actions.append(t2)
        self.limits.append(lin_vel_limit)
        self.cas_vel.append(vel)
        self.joint_poses.append(self.observation[0:6])
        self.real_vels.append(self.observation[48:54])
        self.human_poses.append(self.observation[6:48])
        self.nn_time.append(elapsed_nn)
        self.goals.append(self.goal)
        self.nn_lin_vels.append(nlv)
        self.casadi_lin_vels.append(clv)
        self.minimum_dist.append(min_dist)
        self.ctp.append(T)
        self.file_n.append(self.observation[54])
        self.time.append(elapsed_time)

    def reset(self):
        if self.first<1:
            self.first+=1
            self.init_log_variables()
            self.t_total = time.time()
            set_init_pose(self.start[0:6],6)
            time.sleep(0.5)
            
        else:
            # time.sleep(0.5)
            self.hello_str[1] = 1
            pub_data = Float64MultiArray()
            pub_data.data = self.hello_str
            self.flag_pub.publish(pub_data)
            self.save_log(self.i)
            self.init_log_variables()
            self.threshold_1 = 0.18
            self.threshold_2 = 0.02
            self.hello_str[0]+= 1
            self.hello_str[1] = 0
            pub_data.data = self.hello_str
            self.flag_pub.publish(pub_data)
            # time.sleep(1)
            self.i+=1
        self.d_time_start = time.time()
        self.step()
        
    def init_log_variables(self):
        self.nn_actions = []
        self.joint_poses = []
        self.human_poses = []
        self.real_vels = []
        self.nn_time = []
        self.observation = [1]*55
        self.casadi_lin_vels = []
        self.nn_lin_vels = []
        self.minimum_dist = []
        self.smallest_dist = 100
        self.tp = [0 for c in range(21)]
        self.goals = []
        self.arrive = False
        self.file_n=[]
        self.time = []
        self.ctp = []
        self.cas_vel = []
        self.limits = []

    def save_log(self,save_iter):
        rec_dir = '/home/robot/workspaces/Big_Data/'
        os.chdir(rec_dir)
        write_mat('Tests/loop/' +self.controller + "/"+ self.run_name,
                        {'actions': self.nn_actions,
                        'joint_positions': self.joint_poses,
                        'human_poses':self.human_poses,
                        'real_vels': self.real_vels,
                        'goal':self.goals,
                        'min_dist': self.minimum_dist,
                        'cas_lin_vel':self.casadi_lin_vels,
                        'nn_lin_vel':self.nn_lin_vels,
                        'ctp': self.ctp,
                        'cas_vel':self.cas_vel,
                        'nn_time':self.nn_time,
                        'network':self.nn_layers,
                        'limits':self.limits,
                        'file_n':self.file_n,
                        'time':self.time},
                        str(save_iter))    
        
if __name__ == '__main__':
    rospy.init_node("dagger", anonymous=True)
    run_name = stream_tee.generate_timestamp()
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = [48,36,27]
    
    controller = 'DA-DNN'
    human_mode = 'walk'
    model = MyModel(dev,48,6,n).to(dev)
    print(human_mode)
    env = ENV(run_name,dev,model,n,controller,human_mode)
    t = time.time()
    i = 0
    rate = rospy.Rate(20)
    env.choose_model()
    env.reset()
    while not rospy.is_shutdown():
        done = env.done()
        if done==True:
            i+=1
            env.reset()
            elapsed = time.time() - t
            print("Episode ", i, ' time = ', elapsed)
            t = time.time()
        else:
            env.step()
        rate.sleep()

