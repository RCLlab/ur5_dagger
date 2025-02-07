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

class MyModel(nn.Module):
    def __init__(self, dev, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.dev = dev
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        current_dim = input_dim
        self.layers = nn.ModuleList()
        # self.dropout = nn.Dropout(0.25)
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim,hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim,output_dim))
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
            # x = self.dropout(x)
        out = self.layers[-1](x)
        return out

class ENV:
    def __init__(self,run_name,dev,model,n, direction):
        rospy.Subscriber('/info', Float64MultiArray, self.callback)
        self.pub = rospy.Publisher('/ur_driver/URScript', String, queue_size=1)
        self.flag_pub = rospy.Publisher('/flag', Float64MultiArray, queue_size=1)
        self.run_name = run_name
        self.A = [0.0, -2.3, -1.1, -1.2, -1.2, 0.5]
        self.B = [3.0, -1.6, -1.7, -1.7, -1.7, 1.0]
        self.direction = direction
        if self.direction=='AB':
            self.start = self.A
            self.goal = self.B
        else:
            self.start = self.B
            self.goal = self.A
        self.i = 0
        self.first = 0
        self.hello_str = [-1,0,0]
        self.init_log_variables()
        self.total = 0
        self.nSteps = 0
        self.nTotal = 300
        self.dev = dev
        self.model = model
        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.loss_function = nn.MSELoss()
        self.env_D = []
        self.nn_layers = n
        self.n_batch = 1000
        self.min_loss = 1
        self.total_episode_numbers = 2030
        self.eval_D = []
        self.test_S = []
        self.test_W = []
        self.temp_len = 3000

    def callback(self, data):
        self.observation = data.data[0:169]

    def done(self):
        d = False
        if self.nSteps>self.nTotal-1:
            d = True
        return d
    
    def limit_check(self):
        self.lim = 0
        limit = np.array([[-6.0, 6.0],[-3.14, 3.14],[-3.14, 3.14],[-3.14, 0],[-3.14, 3.14],[-6.0, 6.0]])
        for i in range(6):
            if self.observation[i]<limit[i,0]:
                self.lim = 1
            if self.observation[i]>limit[i,1]: 
                self.lim = 1
                
    def train(self):
        self.runLoss = 0
        self.model.train()
        random.shuffle(self.env_D)
        x_train = np.array(self.env_D)[:,0:48]
        y_train = np.array(self.env_D)[:,48:54]
        for b in range(0, self.temp_len , self.n_batch):
            seq_data = np.array(x_train[b:b+self.n_batch])
            seq_label = np.array(y_train[b:b+self.n_batch])
            seq_data = torch.tensor([i for i in seq_data], dtype=torch.float32).to(dev)
            seq_label = torch.tensor([i for i in seq_label], dtype=torch.float32).to(dev)
            self.optimizer.zero_grad()
            y_pred = self.model(seq_data)
            single_loss = self.loss_function(y_pred, seq_label)
            self.runLoss += single_loss.item()
            single_loss.backward()
            self.optimizer.step()
        self.runLoss /=self.temp_len
        print ('Train loss: {:.8f}, len {}'.format(self.runLoss, len(x_train)))
    

    def evals(self, data, moded):
        total_loss = 0
        model.eval()
        random.shuffle(data)
        x_eval = np.array(data)[:,0:48]
        y_eval = np.array(data)[:,48:54]
        with torch.no_grad():
            for b in range(0, self.temp_len , self.n_batch):
                seq_data = np.array(x_eval[b:b+self.n_batch])
                seq_label = np.array(y_eval[b:b+self.n_batch])
                seq_data = torch.tensor([i for i in seq_data], dtype=torch.float32).to(dev)
                seq_label = torch.tensor([i for i in seq_label], dtype=torch.float32).to(dev)
                y_pred = self.model(seq_data)
                single_loss = self.loss_function(y_pred, seq_label)
                total_loss += single_loss.item()
            total_loss /= self.temp_len
        
        if moded=='validation':
            self.Eval_loss = total_loss
        if moded=='test sit':
            self.Test_s_loss = total_loss
        print('valid loss: {:.8f}, len {}'.format(total_loss, len(x_eval)))
        return total_loss

    def test(self, x_test):
        self.model.eval()
        with torch.no_grad():
            Data = np.array(x_test)
            Data = torch.tensor([i for i in Data], dtype=torch.float32).to(dev)
            Nov = self.model(Data)
        return Nov

    def controller_type(self, nEpisodes, MPC_episodes):
        self.MPC_episodes = MPC_episodes
        self.controller = 'MPC'
        if nEpisodes>MPC_episodes-1:
            self.controller = 'DIL'
        return self.controller

    def step(self):
        t_nn = time.time()
        u = self.test(self.observation[0:48])
        elapsed_nn = time.time() - t_nn
        temp = list(u.cpu().numpy())
        dil_vel = [temp[0],temp[1],temp[2],temp[3],temp[4],temp[5]]
        mpc_vel = [self.observation[48],self.observation[49],self.observation[50],self.observation[51],self.observation[52],self.observation[53]]
        self.limit_check()
        if self.lim==0:
            if self.controller == 'MPC':
                vel = mpc_vel
            else:
                vel = dil_vel
        else:
            vel = [0,0,0,0,0,0]
        hello_str = "speedj(["+str(vel[0])+","+str(vel[1])+","+str(vel[2])+","+str(vel[3])+","+str(vel[4])+","+str(vel[5])+"],"+"5.0"+",0.1)" 
        self.pub.publish(hello_str)
        self.nSteps+=1
        d_time = time.time() - self.d_time_start
        
        if self.hello_str[2]==0:
            self.env_D.append(self.observation[0:54])
        if self.hello_str[2]==1:
            self.eval_D.append(self.observation[0:54])
        if self.hello_str[2]==2:
            self.test_S.append(self.observation[0:54])

        # save data
        self.joint_poses.append(self.observation[0:6])
        self.human_poses.append(self.observation[6:48])
        self.mpc_sol.append(mpc_vel)
        self.dil_sol.append(dil_vel)
        self.ctp.append(self.observation[54:75])
        self.low_goal.append(self.observation[75:81])
        self.real_vels.append(self.observation[81:87])
        self.low_mpc_details.append(self.observation[87:90])
        self.minimum_dist.append(self.observation[90:97])
        self.smallest_dist.append(self.observation[97])
        self.lin_vel_scale.append(self.observation[98])
        self.from_high_controller.append(self.observation[99:129])
        self.goal = self.observation[117:123]
        self.ctv.append(self.observation[129:150])
        self.lin_vel_limit.append(self.observation[150:157])
        self.file_n.append(self.observation[157])
        self.mpc_time.append(self.observation[158])
        self.ctv_linear.append(self.observation[159:166])
        self.max_diff = self.observation[166]
        self.file_start.append(self.observation[167])
        self.mpc_solve_time.append(self.observation[168])
        self.nn_time.append(elapsed_nn)
        self.dagger_time.append(d_time)
        self.nn_sol.append(dil_vel)

    def reset(self):
        if self.first<1:
            self.first+=1
            set_init_pose(self.start[0:6],6)
            time.sleep(2)
            self.init_log_variables()
            self.temp_time = time.time()
            self.temp_mode = 0
        else:
            time.sleep(0.5)
            self.hello_str[1] = 1
            pub_data = Float64MultiArray()
            pub_data.data = self.hello_str
            self.flag_pub.publish(pub_data)
            self.save_log(self.i)
            t1 = np.array(self.observation)
            t2 = np.array(self.start)
            b = np.array((t1[0:6] - t2[0:6]))
            dist_target = np.linalg.norm(b)
            t = int(dist_target)
            print("Initialize the robot in ",t,"s")
            set_init_pose(self.start[0:6],t)
            time.sleep(0.5)
            self.init_log_variables()
            if self.temp_mode>1:
                self.temp_mode=0
                self.hello_str[2]=0
            else:
                self.temp_mode+=1
                self.hello_str[2]+=1
            self.hello_str[0]+= 1
            self.hello_str[1] = 0
            pub_data.data = self.hello_str
            self.flag_pub.publish(pub_data)
            time.sleep(1)
        self.i+=1
        self.d_time_start = time.time()
        self.step()
        
    def init_log_variables(self):
        self.Eval_loss = 1
        self.Test_s_loss = 1
        self.runLoss = 1
        self.lim = 0
        self.nn_loss = 0
        self.observation = [1]*169
        self.joint_poses = []
        self.human_poses = []
        self.mpc_sol = []
        self.ctp = []
        self.low_goal = []
        self.real_vels = []
        self.low_mpc_details = []
        self.minimum_dist = []
        self.smallest_dist = []
        self.lin_vel_scale = []
        self.from_high_controller = []
        self.ctv = []
        self.lin_vel_limit = []
        self.file_n=[]
        self.mpc_time = []
        self.file_start = []
        self.time = []
        self.ctv_linear = []
        self.diff = 10
        self.mpc_solve_time = []
        self.nn_time = []
        self.dagger_time = []
        self.nSteps = 0
        self.nn_sol = []
        self.dil_sol = []
        
    def save_log(self,save_iter):
        rec_dir = '/home/robot/workspaces/Big_Data/'
        os.chdir(rec_dir)
        Total_time = time.time() - self.temp_time
        write_mat('dagger/' + str(self.MPC_episodes) + '/' + self.direction+'/'+self.run_name,
                        {'ctp':self.ctp,
                        'joint_positions': self.joint_poses,
                        'human_poses':self.human_poses,
                        'mpc_sol':  self.mpc_sol,
                        'real_vels': self.real_vels,
                        'low_goal': self.low_goal,
                        'dil_sol':self.dil_sol,
                        'minimum_distance': self.minimum_dist,
                        'Eval_loss':self.Eval_loss,
                        'Test_s_loss':self.Test_s_loss,
                        'low_mpc_details': self.low_mpc_details,
                        'smallest_dist': self.smallest_dist,
                        'lin_vel_scale':self.lin_vel_scale,
                        'from_high_controller':self.from_high_controller,
                        'ctv': self.ctv,
                        'lin_vel_limit': self.lin_vel_limit,
                        'mpc_time': self.mpc_time,
                        'file_n':self.file_n,
                        'file_start':self.file_start,
                        'time':self.time,
                        'nn_time':self.nn_time,
                        'dagger_time':self.dagger_time,
                        'mpc_solve_time':self.mpc_solve_time,
                        'nn_loss':self.runLoss,
                        'nn_sol':self.nn_sol,
                        'MPC_episodes': self.MPC_episodes,
                        'total_time':Total_time,
                        'nSteps':self.nTotal,
                        'nn_layers':self.nn_layers,
                        'learning_rate':self.lr,
                        'ctv_linear': self.ctv_linear},
                        str(save_iter))    
        
        if save_iter<self.total_episode_numbers and self.Eval_loss<self.min_loss:
            self.min_loss = self.Eval_loss
            print("\nweights are saved with min loss ", self.min_loss)
            os.chdir(rec_dir+'dagger/' + str(self.MPC_episodes) + '/' + self.direction+'/'+self.run_name,)
            torch.save(model.state_dict(),"model.pth")

if __name__ == '__main__':
    rospy.init_node("dagger", anonymous=True)
    run_name = stream_tee.generate_timestamp()
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = [48,36,27]
    model = MyModel(dev,48,6,n).to(dev)
    direction = 'toA'
    env = ENV(run_name,dev,model,n,direction)
    i = 0
    rate = rospy.Rate(20)
    MPC_controller = 'MPC'
    DIL_controller = 'DIL'
    MPC_episodes = 30
    
    controller = env.controller_type(i, MPC_episodes)
    env.reset()

    while not rospy.is_shutdown():
        done = env.done()
        if done==True:
            print("Done:", env.hello_str[2], len(env.env_D),len(env.eval_D),len(env.test_S))
            if i>29:
                env.train()
                env.evals(env.eval_D, 'validation')
                env.evals(env.test_S, 'test sit')
            i+=1
            controller = env.controller_type(i, MPC_episodes)
            print("\n", controller, "episode", i)
            env.reset()
        else:
            env.step()
        rate.sleep()

    
        
