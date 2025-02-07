#!/usr/bin/env python
import os
import stream_tee as stream_tee
# import __main__ as main
import rospy
from std_msgs.msg import Float64MultiArray, String
import time
from stream_tee import write_mat
# from initialization import set_init_pose

from cascadi_solver import cascadi_solv
class ENV:
    def __init__(self,run_name):
        rospy.Subscriber('/info', Float64MultiArray, self.callback)
        self.pub = rospy.Publisher('/ur_driver/URScript', String, queue_size=1)
        self.flag_pub = rospy.Publisher('/flag', Float64MultiArray, queue_size=1)
        self.run_name = run_name
        self.A = [0.0, -2.3, -1.1, -1.2, -1.2, 0.5]
        self.goal = self.A
        self.B = [3.0, -1.6, -1.7, -1.7, -1.7, 1.0]
        self.i = 0
        self.first = 0
        self.init_log_variables()
        self.total = 0
        self.hello_str=[-1,0]
        self.t_total = time.time()
        self.max_diff = 10

    def callback(self, data):
        self.observation = data.data[0:169]

    def done(self):
        arrive = False
        if self.goal[0]==self.A[0]:
            if self.max_diff<0.02:
                arrive = True
        return arrive

    def step(self):
        vel = [self.observation[48],self.observation[49],self.observation[50],self.observation[51],self.observation[52],self.observation[53]]
        hello_str = "speedj(["+str(vel[0])+","+str(vel[1])+","+str(vel[2])+","+str(vel[3])+","+str(vel[4])+","+str(vel[5])+"],"+"5.0"+",0.1)" 
        elapsed_time = time.time() - self.t_total
        max_vell, lin_vell_limit_arr, ctp, min_dist = cascadi_solv(vel,self.observation[0:48])
        self.pub.publish(hello_str)
        self.joint_poses.append(self.observation[0:6])
        self.human_poses.append(self.observation[6:48])
        self.mpc_sol.append(vel)
        self.ctp.append(ctp)
        self.low_goal.append(self.observation[75:81])
        self.real_vels.append(self.observation[81:87])
        self.low_mpc_details.append(self.observation[87:90])
        self.minimum_dist.append(min_dist)
        self.smallest_dist.append(self.observation[97])
        self.lin_vel_scale.append(self.observation[98])
        self.from_high_controller.append(self.observation[99:129])
        self.goal = self.observation[117:123]
        self.ctv.append(self.observation[129:150])
        self.lin_vel_limit.append(lin_vell_limit_arr)
        self.file_n.append(self.observation[157])
        self.mpc_time.append(self.observation[158])
        self.ctv_linear.append(max_vell)
        self.max_diff = self.observation[166]
        self.file_start.append(self.observation[167])
        self.mpc_solve_time.append(self.observation[168])
        self.time.append(elapsed_time)
    
    def reset(self):
        if self.first<2:
            self.first+=1
            self.init_log_variables()
            self.t_total = time.time()
        else:
            time.sleep(0.5)
            print("goal", self.goal)
            self.hello_str[1] = 1
            pub_data = Float64MultiArray()
            pub_data.data = self.hello_str
            self.flag_pub.publish(pub_data)
            self.save_log(self.i)
            time.sleep(2)
            self.init_log_variables()
            self.hello_str[0]+= 1
            self.hello_str[1] = 0
            pub_data.data = self.hello_str
            self.flag_pub.publish(pub_data)
            time.sleep(1)
            self.i+=1
        self.step()
    
    def init_log_variables(self):
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

    def save_log(self,save_iter):
        rec_dir = '/home/robot/workspaces/Big_Data/'
        os.chdir(rec_dir)
        write_mat('dagger_tests/Loop/MPC/125/' + self.run_name,
                        {'ctp':self.ctp,
                        'joint_positions': self.joint_poses,
                        'human_poses':self.human_poses,
                        'mpc_sol':  self.mpc_sol,
                        'real_vels': self.real_vels,
                        'low_goal': self.low_goal,
                        'minimum_distance': self.minimum_dist,
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
                        'mpc_solve_time':self.mpc_solve_time,
                        'ctv_linear': self.ctv_linear},
                        str(save_iter))    

if __name__ == '__main__':
    rospy.init_node("mpc_test", anonymous=True)
    run_name = stream_tee.generate_timestamp()
    t = time.time()
    i = 0
    save_iter = 0
    rate = rospy.Rate(20) #hz
    env = ENV(run_name)
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

    
        
