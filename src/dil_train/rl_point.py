import pybullet as pb
import pybullet_utils.bullet_client as bc
import gym
from gym import spaces
from gym.utils import seeding
import time as clock
import numpy as np
import pybullet_data
from stream_tee import write_mat
import os
import check_distance
import ssm

alpha = 0

class ur5DRL(gym.Env):
    def __init__(self,run_name, start_pose=[1.5, -2.5, -1.1, -1.7, -1.7, 1.0], target_pose=[0, 0, 0, 0, 0, 0], max_episode_length=400, render_mode=None):
        self.render_mode = render_mode
        self.c1 = 10
        self.c2 = 1
        self.c3 = 45
        self.init_log_variables()
        self.episodes = 0
        self.run_name = run_name
        self.observation = [0]*18
        self.max_episode_length = max_episode_length
        self.table_z = 0.625

        self.target_pose = target_pose

        self.target_position, min_dist = check_distance.get_end_effector_pos((self.target_pose[0], self.target_pose[1], self.target_pose[2], self.target_pose[3], self.target_pose[4], self.target_pose[5]), [0, 0, 0])
        # self.start_pose = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
        self.start_pose = start_pose

        # self.start_pose = self.target_pose

        self._start()

    def _start(self):
        if self.render_mode == "human":
            # p.connect(p.GUI) # or p.DIRECT for non-graphical version
            self.p = bc.BulletClient(connection_mode=pb.GUI)
        else: 
            # p.connect(p.DIRECT)
            self.p = bc.BulletClient(connection_mode=pb.DIRECT)
        
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        self.p.setGravity(0,0,0)
        # p.setTimeStep(1./240.)
        # p.setRealTimeSimulation(0)
        # planeId = self.p.loadURDF("plane.urdf")
        startPos = [0,0, 0]
        self.sphere_joint_pos = [0, 0, 0.5]
        startOrientation = self.p.getQuaternionFromEuler([0,0,0])
        self.sphere_joint_radius = 0.6
        startOrientation2 = self.p.getQuaternionFromEuler([0,0, 0])
        table = self.p.loadURDF("/home/robot/RL_pybullet/table/table.urdf", [0, 0, -self.table_z], startOrientation)
        self.sphere = self.p.loadURDF("/home/robot/RL_pybullet/prismatic_robot.urdf", self.sphere_joint_pos, startOrientation2)
        self.UR5 = self.p.loadURDF("/home/robot/RL_pybullet/full_UR5.urdf", startPos, startOrientation)

        self.joint1_limits = np.deg2rad([-360, 360])
        self.joint2_limits = np.deg2rad([-360, 360])
        self.joint3_limits = np.deg2rad([-180, 180])
        self.joint4_limits = np.deg2rad([-360, 360])
        self.joint5_limits = np.deg2rad([-360, 360])
        self.joint6_limits = np.deg2rad([-360, 360])

        self.start_offset = np.random.randint(-33, 33)

        self.sphere_pos = [0, 0, 0]
        self.prev_sphere_pos = self.sphere_pos
        self.p.resetJointStateMultiDof(self.sphere, 0, targetValue=[0], targetVelocity=[0.0])
        mode = self.p.POSITION_CONTROL
        self.p.setJointMotorControlArray(self.sphere, [0],
                controlMode=mode, targetPositions = [np.sin((self.start_offset)/40)*1.5],
                forces = [25.0])

        self.p.resetJointStateMultiDof(self.UR5, 1, targetValue=[self.start_pose[0]], targetVelocity=[0.0])
        self.p.resetJointStateMultiDof(self.UR5, 2, targetValue=[self.start_pose[1]], targetVelocity=[0.0])
        self.p.resetJointStateMultiDof(self.UR5, 3, targetValue=[self.start_pose[2]], targetVelocity=[0.0])
        self.p.resetJointStateMultiDof(self.UR5, 4, targetValue=[self.start_pose[3]], targetVelocity=[0.0])
        self.p.resetJointStateMultiDof(self.UR5, 5, targetValue=[self.start_pose[4]], targetVelocity=[0.0])
        self.p.resetJointStateMultiDof(self.UR5, 6, targetValue=[self.start_pose[5]], targetVelocity=[0.0])

        # Init of state space (observations). This is needed for Gym/Learning algorithm.

        self.obs_max = np.array([self.joint1_limits[1],self.joint2_limits[1],self.joint3_limits[1],self.joint4_limits[1],self.joint5_limits[1],self.joint6_limits[1],
                                 1, 1, 1, 1, 1, 1, np.inf, np.inf, np.inf,
                                 np.inf, np.inf, np.inf
                                 ]).astype('float32')
        self.obs_min = np.array([self.joint1_limits[0],self.joint2_limits[0],self.joint3_limits[0],self.joint4_limits[0],self.joint5_limits[0],self.joint6_limits[0],
                                 -1, -1, -1, -1, -1, -1, -np.inf, -np.inf, -np.inf,
                                 -np.inf, -np.inf, -np.inf,
                                 ]).astype('float32')

        self.act_max = np.array([1,1,1,1,1,1]).astype('float32') # rad/s
        self.act_min = np.array([-1,-1,-1,-1,-1,-1]).astype('float32') # rad/s

        self.action_space = spaces.Box(self.act_min, self.act_max)
        self.observation_space = spaces.Box(self.obs_min, self.obs_max)

        self.seed()
        self.first_reset = 0
        self._envStepCounter = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        global alpha

        u = np.clip(u, self.act_min, self.act_max)
        for _ in range(12):
            self.send_actions(u)
            self.p.stepSimulation()
        # clock.sleep(1./240.)
        
        mode = self.p.POSITION_CONTROL
        self.p.setJointMotorControlArray(self.sphere, [0],
                controlMode=mode, targetPositions = [np.sin((self._envStepCounter + self.start_offset)/40)*1.5],
                forces = [25.0])
        sphere_state = self.p.getJointState(self.sphere, 0)[0]/2
        # print(self.sphere_pos)
        # print(np.rad2deg(theta))
        self.prev_sphere_pos = self.sphere_pos
        self.sphere_pos = [self.sphere_joint_pos[0] + sphere_state, self.sphere_joint_pos[1], self.sphere_joint_pos[2]]
        # print(self.sphere_pos)

        self.self_observe()
        self._envStepCounter += 1
        reward, r_target, r_action, r_obstacle_sphere, gammas = self.reward(u)

        done = False
        if self._envStepCounter > self.max_episode_length: done = True

        return self.observation, reward, done, dict(reward_dist=r_target,reward_ctrl=r_action, gammas=gammas)

    def self_observe(self):
        states = self.p.getJointStates(self.UR5,[1,2,3,4,5,6])
        jp = [x[0] for x in states]
        jv = [x[1] for x in states]

        curr_end_effector_pos, min_dist = check_distance.get_end_effector_pos((jp[0], jp[1], jp[2], jp[3], jp[4], jp[5]), self.sphere_pos)


        self.observation = np.array([jp[0],jp[1],jp[2],jp[3],jp[4],jp[5],
                                     jv[0],jv[1],jv[2],jv[3],jv[4],jv[5],
                                     curr_end_effector_pos[0], curr_end_effector_pos[1], curr_end_effector_pos[2],
                                     self.sphere_pos[0], self.sphere_pos[1], self.sphere_pos[2],
                                     ]).astype('float32')
    

    def reset(self):
        self.start_offset = np.random.randint(-33, 33)
        self.sphere_pos = [0, 0, 0]
        self.prev_sphere_pos = self.sphere_pos
        self.p.resetJointStateMultiDof(self.sphere, 0, targetValue=[0], targetVelocity=[0.0])
        mode = self.p.POSITION_CONTROL
        self.p.setJointMotorControlArray(self.sphere, [0],
                controlMode=mode, targetPositions = [np.sin((self.start_offset)/40)*1.5],
                forces = [25.0])

        self.p.resetJointStateMultiDof(self.UR5, 1, targetValue=[self.start_pose[0]], targetVelocity=[0.0])
        self.p.resetJointStateMultiDof(self.UR5, 2, targetValue=[self.start_pose[1]], targetVelocity=[0.0])
        self.p.resetJointStateMultiDof(self.UR5, 3, targetValue=[self.start_pose[2]], targetVelocity=[0.0])
        self.p.resetJointStateMultiDof(self.UR5, 4, targetValue=[self.start_pose[3]], targetVelocity=[0.0])
        self.p.resetJointStateMultiDof(self.UR5, 5, targetValue=[self.start_pose[4]], targetVelocity=[0.0])
        self.p.resetJointStateMultiDof(self.UR5, 6, targetValue=[self.start_pose[5]], targetVelocity=[0.0])

        write_mat('../log_reward/' + self.run_name,
                    {'q_dot': self.q_dot,
                    'r_target': self.r_target,
                    'r_vel': self.r_vel,
                    'r_obstacle': self.r_obstacle,
                    # 'r_obstacle_sphere': self.r_obstacle_sphere,
                    'q': self.q,
                    'target':self.target_pose,
                    'r_tot': self.r_tot},
                    str(self.episodes))

        self.episodes+=1
        self.init_log_variables()
        for i in range(100):
            self.p.stepSimulation()
        self.self_observe()
        return self.observation

    def reward(self, u):
        b = np.array((self.observation[0:6] - self.target_pose[0:6]))
        dist_target = np.linalg.norm(b)
        # if dist_target < 0.1:
        #     r_target_dist = -(0.1 * (dist_target ** 2))
        # else:
        #     r_target_distr_obstacle_table, r_obstacle_spherenalg.norm(u) ** 2)

        r_target_dist = -(dist_target ** 2)
        r_target_dist *= self.c1
        r_action = - (np.linalg.norm(u) ** 2)
        r_action *= self.c2

        q = (self.observation[0], self.observation[1], self.observation[2], self.observation[3], self.observation[4], self.observation[5])
        distance_to_table = check_distance.get_min_distance_to_table(q)
        # print(distance_to_table)
        distance_to_sphere, gammas = ssm.calculate_ssm(self.observation[0:6], self.observation[6:12], [self.sphere_pos[0], self.sphere_pos[1], self.sphere_pos[2], 0.1])
        if distance_to_table < 0: distance_to_table = 0
        # r_obstacle_table = -(1 / (distance_to_table + 1))**8
        # r_obstacle_table *= self.c3 * 1.5

        if distance_to_sphere < 0: distance_to_sphere = 0
        # r_obstacle_sphere = -(1 / (distance_to_sphere + 1))**8
        # r_obstacle_sphere *= self.c3

        r_obstacle = -(1 / (min(distance_to_table, distance_to_sphere) + 1))**8 * self.c3

        reward = r_target_dist + r_action + r_obstacle
        self.log_variables(u, r_target_dist, r_action, r_obstacle, self.observation[0:6], reward)
        return reward, r_target_dist, r_action, r_obstacle, gammas

    def send_actions(self, u):
        maxForce = [25.0,25.0,25.0,25.0,25.0,25.0]
        mode = self.p.VELOCITY_CONTROL
        self.p.setJointMotorControlArray(self.UR5, [1,2,3,4,5,6],
                controlMode=mode, targetVelocities = u,
                forces = maxForce)

    def init_log_variables(self):
        self.q_dot = [0]*6
        self.r_target = []
        self.r_vel = []
        self.r_obstacle = []
        # self.r_obstacle_sphere = []
        self.q = [0]*6
        self.r_tot = []
        self._envStepCounter = 0

    def log_variables(self, q_dot, r_target, r_vel, r_obstacle, q, r_tot):
        self.q_dot = np.vstack([self.q_dot, q_dot])
        self.q = np.vstack([self.q, q])
        self.r_target.append(r_target)
        self.r_vel.append(r_vel)
        self.r_obstacle.append(r_obstacle)
        self.r_tot.append(r_tot)
