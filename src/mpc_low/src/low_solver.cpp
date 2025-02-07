#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Float64.h>
#include <std_msgs/String.h>
#include <sensor_msgs/JointState.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <eigen3/Eigen/Dense>
#include <math.h>
#include <std_msgs/Int32.h>
#include <gazebo_msgs/SetModelConfiguration.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <rosgraph_msgs/Clock.h>
#include "acados_solver_holder.cpp"
#include <visualization_msgs/Marker.h>

using namespace std;

float dist_v(Eigen::Vector3f v, Eigen::Vector3f w){
	return (v-w).norm();
}

double z_sh = 0.1;
Eigen::MatrixXf get_cpose(float theta_1, float theta_2, float theta_3, float theta_4, float theta_5, float theta_6){
Eigen::MatrixXf mat(3,8);
mat << 0, 0.06*sin(theta_1), (-0.425*cos(theta_1)*cos(theta_2))/2+0.14*sin(theta_1), -0.425*cos(theta_1)*cos(theta_2)+0.11*sin(theta_1), -0.425*cos(theta_1)*cos(theta_2)+(-(cos(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2)))/4000-(-0.425*cos(theta_1)*cos(theta_2)))/3+0.02*sin(theta_1), -0.425*cos(theta_1)*cos(theta_2)+2*(-(cos(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2)))/4000-(-0.425*cos(theta_1)*cos(theta_2)))/3+0.02*sin(theta_1), -(cos(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2)))/4000+0.06*sin(theta_1), 0.10915*sin(theta_1) - 0.425*cos(theta_1)*cos(theta_2) + 0.0823*cos(theta_5)*sin(theta_1) + 0.39225*cos(theta_1)*sin(theta_2)*sin(theta_3) - 0.0823*cos(theta_2 + theta_3 + theta_4)*cos(theta_1)*sin(theta_5) + 0.09465*cos(theta_2 + theta_3)*cos(theta_1)*sin(theta_4) + 0.09465*sin(theta_2 + theta_3)*cos(theta_1)*cos(theta_4) - 0.39225*cos(theta_1)*cos(theta_2)*cos(theta_3)-0.05*sin(theta_1),
       0,-0.06*cos(theta_1), (-0.425*cos(theta_2)*sin(theta_1))/2-0.14*cos(theta_1), -0.425*cos(theta_2)*sin(theta_1)-0.11*cos(theta_1), -0.425*cos(theta_2)*sin(theta_1)+(-(sin(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2)))/4000-(-0.425*cos(theta_2)*sin(theta_1)))/3-0.02*cos(theta_1), -0.425*cos(theta_2)*sin(theta_1)+2*(-(sin(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2)))/4000-(-0.425*cos(theta_2)*sin(theta_1)))/3-0.02*cos(theta_1), -(sin(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2)))/4000-0.06*cos(theta_1), 0.39225*sin(theta_1)*sin(theta_2)*sin(theta_3) - 0.0823*cos(theta_1)*cos(theta_5) - 0.425*cos(theta_2)*sin(theta_1) - 0.10915*cos(theta_1) - 0.0823*cos(theta_2 + theta_3 + theta_4)*sin(theta_1)*sin(theta_5) + 0.09465*cos(theta_2 + theta_3)*sin(theta_1)*sin(theta_4) + 0.09465*sin(theta_2 + theta_3)*cos(theta_4)*sin(theta_1) - 0.39225*cos(theta_2)*cos(theta_3)*sin(theta_1)+0.05*cos(theta_1),
       0, 0.0894+z_sh,            (0.0894 - 0.425*sin(theta_2))/2+z_sh,                        0.0894 - 0.425*sin(theta_2)+z_sh,                       0.0894 - 0.425*sin(theta_2)+(0.0894 - 0.425*sin(theta_2) - 0.39225*sin(theta_2 + theta_3)-(0.0894 - 0.425*sin(theta_2)))/3+z_sh,                                            0.0894 - 0.425*sin(theta_2)+2*(0.0894 - 0.425*sin(theta_2) - 0.39225*sin(theta_2 + theta_3)-(0.0894 - 0.425*sin(theta_2)))/3+z_sh,                                            0.0894 - 0.425*sin(theta_2) - 0.39225*sin(theta_2 + theta_3)+z_sh,                                 0.09465*sin(theta_2 + theta_3)*sin(theta_4) - 0.425*sin(theta_2) - 0.39225*sin(theta_2 + theta_3) - sin(theta_5)*(0.0823*cos(theta_2 + theta_3)*sin(theta_4) + 0.0823*sin(theta_2 + theta_3)*cos(theta_4)) - 0.09465*cos(theta_2 + theta_3)*cos(theta_4) + 0.08945+z_sh;
	return mat;
}

Eigen::MatrixXf get_velocity(float theta_1, float theta_2, float theta_3, float theta_4, float theta_5, float theta_6, 
                             float u_1, float u_2, float u_3, float u_4, float u_5, float u_6){
	Eigen::MatrixXf mat(21,1);
  mat << 0.06*u_1*cos(theta_1),
        0.06*u_1*sin(theta_1),
        0,
        u_1*(0.14*cos(theta_1) + 0.2125*cos(theta_2)*sin(theta_1)) + 0.2125*u_2*cos(theta_1)*sin(theta_2),
        u_1*(0.14*sin(theta_1) - 0.2125*cos(theta_1)*cos(theta_2)) + 0.2125*u_2*sin(theta_1)*sin(theta_2),
        -0.2125*u_2*cos(theta_2),
        u_1*(0.11*cos(theta_1) + 0.425*cos(theta_2)*sin(theta_1)) + 0.425*u_2*cos(theta_1)*sin(theta_2),
        u_1*(0.11*sin(theta_1) - 0.425*cos(theta_1)*cos(theta_2)) + 0.425*u_2*sin(theta_1)*sin(theta_2),
        -0.425*u_2*cos(theta_2),
        0.02*u_1*cos(theta_1) + 0.13075*u_1*cos(theta_2 + theta_3)*sin(theta_1) + 0.13075*u_2*sin(theta_2 + theta_3)*cos(theta_1) + 0.13075*u_3*sin(theta_2 + theta_3)*cos(theta_1) + 0.425*u_1*cos(theta_2)*sin(theta_1) + 0.425*u_2*cos(theta_1)*sin(theta_2),
        0.02*u_1*sin(theta_1) - 0.13075*u_1*cos(theta_2 + theta_3)*cos(theta_1) + 0.13075*u_2*sin(theta_2 + theta_3)*sin(theta_1) + 0.13075*u_3*sin(theta_2 + theta_3)*sin(theta_1) - 0.425*u_1*cos(theta_1)*cos(theta_2) + 0.425*u_2*sin(theta_1)*sin(theta_2),
        - 1.0*u_2*(0.13075*cos(theta_2 + theta_3) + 0.425*cos(theta_2)) - 0.13075*u_3*cos(theta_2 + theta_3),
        0.02*u_1*cos(theta_1) + 0.2615*u_1*cos(theta_2 + theta_3)*sin(theta_1) + 0.2615*u_2*sin(theta_2 + theta_3)*cos(theta_1) + 0.2615*u_3*sin(theta_2 + theta_3)*cos(theta_1) + 0.425*u_1*cos(theta_2)*sin(theta_1) + 0.425*u_2*cos(theta_1)*sin(theta_2),
        0.02*u_1*sin(theta_1) - 0.2615*u_1*cos(theta_2 + theta_3)*cos(theta_1) + 0.2615*u_2*sin(theta_2 + theta_3)*sin(theta_1) + 0.2615*u_3*sin(theta_2 + theta_3)*sin(theta_1) - 0.425*u_1*cos(theta_1)*cos(theta_2) + 0.425*u_2*sin(theta_1)*sin(theta_2),
        - 1.0*u_2*(0.2615*cos(theta_2 + theta_3) + 0.425*cos(theta_2)) - 0.2615*u_3*cos(theta_2 + theta_3),
        u_1*(0.06*cos(theta_1) + 0.00025*sin(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2))) + 0.00025*u_2*cos(theta_1)*(1569.0*sin(theta_2 + theta_3) + 1700.0*sin(theta_2)) + 0.39225*u_3*sin(theta_2 + theta_3)*cos(theta_1),
        u_1*(0.06*sin(theta_1) - 0.00025*cos(theta_1)*(1569.0*cos(theta_2 + theta_3) + 1700.0*cos(theta_2))) + 0.00025*u_2*sin(theta_1)*(1569.0*sin(theta_2 + theta_3) + 1700.0*sin(theta_2)) + 0.39225*u_3*sin(theta_2 + theta_3)*sin(theta_1),
        - 1.0*u_2*(0.39225*cos(theta_2 + theta_3) + 0.425*cos(theta_2)) - 0.39225*u_3*cos(theta_2 + theta_3),
        0.05915*u_1*cos(theta_1) + 0.0823*u_1*cos(theta_1)*cos(theta_5) + 0.425*u_1*cos(theta_2)*sin(theta_1) + 0.425*u_2*cos(theta_1)*sin(theta_2) - 0.0823*u_5*sin(theta_1)*sin(theta_5) + 0.09465*u_2*cos(theta_2 + theta_3)*cos(theta_1)*cos(theta_4) + 0.09465*u_3*cos(theta_2 + theta_3)*cos(theta_1)*cos(theta_4) + 0.09465*u_4*cos(theta_2 + theta_3)*cos(theta_1)*cos(theta_4) - 0.09465*u_1*cos(theta_2 + theta_3)*sin(theta_1)*sin(theta_4) - 0.09465*u_1*sin(theta_2 + theta_3)*cos(theta_4)*sin(theta_1) - 0.09465*u_2*sin(theta_2 + theta_3)*cos(theta_1)*sin(theta_4) - 0.09465*u_3*sin(theta_2 + theta_3)*cos(theta_1)*sin(theta_4) - 0.09465*u_4*sin(theta_2 + theta_3)*cos(theta_1)*sin(theta_4) + 0.39225*u_1*cos(theta_2)*cos(theta_3)*sin(theta_1) + 0.39225*u_2*cos(theta_1)*cos(theta_2)*sin(theta_3) + 0.39225*u_2*cos(theta_1)*cos(theta_3)*sin(theta_2) + 0.39225*u_3*cos(theta_1)*cos(theta_2)*sin(theta_3) + 0.39225*u_3*cos(theta_1)*cos(theta_3)*sin(theta_2) - 0.39225*u_1*sin(theta_1)*sin(theta_2)*sin(theta_3) - 0.0823*u_5*cos(theta_2 + theta_3 + theta_4)*cos(theta_1)*cos(theta_5) + 0.0823*u_1*cos(theta_2 + theta_3 + theta_4)*sin(theta_1)*sin(theta_5) + 0.0823*u_2*sin(theta_2 + theta_3 + theta_4)*cos(theta_1)*sin(theta_5) + 0.0823*u_3*sin(theta_2 + theta_3 + theta_4)*cos(theta_1)*sin(theta_5) + 0.0823*u_4*sin(theta_2 + theta_3 + theta_4)*cos(theta_1)*sin(theta_5),
        0.05915*u_1*sin(theta_1) - 0.425*u_1*cos(theta_1)*cos(theta_2) + 0.0823*u_1*cos(theta_5)*sin(theta_1) + 0.0823*u_5*cos(theta_1)*sin(theta_5) + 0.425*u_2*sin(theta_1)*sin(theta_2) + 0.09465*u_1*cos(theta_2 + theta_3)*cos(theta_1)*sin(theta_4) + 0.09465*u_1*sin(theta_2 + theta_3)*cos(theta_1)*cos(theta_4) + 0.09465*u_2*cos(theta_2 + theta_3)*cos(theta_4)*sin(theta_1) + 0.09465*u_3*cos(theta_2 + theta_3)*cos(theta_4)*sin(theta_1) + 0.09465*u_4*cos(theta_2 + theta_3)*cos(theta_4)*sin(theta_1) - 0.39225*u_1*cos(theta_1)*cos(theta_2)*cos(theta_3) - 0.09465*u_2*sin(theta_2 + theta_3)*sin(theta_1)*sin(theta_4) - 0.09465*u_3*sin(theta_2 + theta_3)*sin(theta_1)*sin(theta_4) - 0.09465*u_4*sin(theta_2 + theta_3)*sin(theta_1)*sin(theta_4) + 0.39225*u_1*cos(theta_1)*sin(theta_2)*sin(theta_3) + 0.39225*u_2*cos(theta_2)*sin(theta_1)*sin(theta_3) + 0.39225*u_2*cos(theta_3)*sin(theta_1)*sin(theta_2) + 0.39225*u_3*cos(theta_2)*sin(theta_1)*sin(theta_3) + 0.39225*u_3*cos(theta_3)*sin(theta_1)*sin(theta_2) - 0.0823*u_1*cos(theta_2 + theta_3 + theta_4)*cos(theta_1)*sin(theta_5) - 0.0823*u_5*cos(theta_2 + theta_3 + theta_4)*cos(theta_5)*sin(theta_1) + 0.0823*u_2*sin(theta_2 + theta_3 + theta_4)*sin(theta_1)*sin(theta_5) + 0.0823*u_3*sin(theta_2 + theta_3 + theta_4)*sin(theta_1)*sin(theta_5) + 0.0823*u_4*sin(theta_2 + theta_3 + theta_4)*sin(theta_1)*sin(theta_5),
        u_4*(0.09465*sin(theta_2 + theta_3 + theta_4) - 0.0823*cos(theta_2 + theta_3 + theta_4)*sin(theta_5)) - 1.0*u_3*(0.39225*cos(theta_2 + theta_3) - 0.09465*sin(theta_2 + theta_3 + theta_4) + 0.0823*cos(theta_2 + theta_3 + theta_4)*sin(theta_5)) - 1.0*u_2*(0.39225*cos(theta_2 + theta_3) + 0.425*cos(theta_2) - 0.09465*cos(theta_2 + theta_3)*sin(theta_4) - 0.09465*sin(theta_2 + theta_3)*cos(theta_4) + 0.0823*cos(theta_2 + theta_3)*cos(theta_4)*sin(theta_5) - 0.0823*sin(theta_2 + theta_3)*sin(theta_4)*sin(theta_5)) - 0.0823*u_5*sin(theta_2 + theta_3 + theta_4)*cos(theta_5);
  return mat;
}

// Introduce class to make safer goal change
class GoalFollower 
{ 
    // Access specifier 
    public: 

    // Data Members 
    ros::Publisher info_pub;
    double robot_spheres[7] = {0.15, 0.15, 0.15, 0.08, 0.08, 0.12, 0.1};
    
    double human_sphere[58]= {10.0517,   0.5220,   1.0895,   0.1500,
                              10.0658,   0.4526,   0.8624,   0.2500,
                              10.0844,   0.7044,   0.9207,   0.1500,
                              10.2083,   0.3075,   1.0208,   0.1500,
                              10.0556,   0.6289,   0.7595,   0.1500,
                              10.2024,   0.2732,   0.8478,   0.1500,
                              10.0267,   0.5535,   0.5983,   0.1500,
                              10.1965,   0.2389,   0.6749,   0.1500,
                             -10.0208,   0.3964,   0.5857,   0.1000,
                              10.0546,   0.2951,   0.6132,   0.1000,
                             -10.1062,   0.2444,   0.5897,   0.1300,
                             -10.0998,   0.3062,   0.5387,   0.1300,
                              10.1908,   0.5290,   1.0016,   0.2000,
                              10.2106,   0.4602,   0.6915,   0.2500,
                              0, 0};

    double goal_queue[120] = {0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000, 
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000, 
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000, 
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000,
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000, 
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000, 
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000, 
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000,
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000,
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000,
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000,
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000,
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000,
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000,
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000,
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000,
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000,
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000,
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000,
                              0.0000, -1.57, 0.0000, -1.57, 0.0000, 0.0000};
                              
    double goal[6] = {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000};
    double comand_vel[6] = {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000};
    double joint_position[6] = {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000};
    double joint_speed[6] = {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000};
    double from_high[32] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10};

    // Member Functions() 
    void change_obstacles_msg(const std_msgs::Float64MultiArray obstacle_data){ 
      for (int i=0; i<58; i++) human_sphere[i] = obstacle_data.data[i];
    }

    void change_goal_msg(const std_msgs::Float64MultiArray joint_pose_values) { 
      for (int i=0; i<32; i++) from_high[i]= joint_pose_values.data[i];

      for (int j=0; j<5; j++) {
        for (int i=0; i<6; i++){
          goal_queue[j*6+i] = joint_pose_values.data[i]*(0.050*(j+5)) + joint_pose_values.data[i+12]; 
        }
      }
      for (int j=5; j<15; j++) {
        for (int i=0; i<6; i++){ 
          goal_queue[j*6+i] = joint_pose_values.data[i+6]*(0.050*(j-5)) + goal_queue[4*6+i]; 
        }
      }
      for (int i=0; i<6; i++) goal[i] = goal_queue[i];
      int correction_steps = 0;
      int joint_corrections = 0;
      for (int i = 0; i < 14; i++) {
        joint_corrections = 0;
        for (int j=0; j < 6; j++) {
          if ((joint_position[j] - goal_queue[i*6+j])>(goal_queue[(i+1)*6+j] - goal_queue[i*6+j])) joint_corrections++;
        }
        if (joint_corrections>3) correction_steps++;
      }
    }

    void change_states_msg(const std_msgs::Float64MultiArray::ConstPtr& msg){
      for (int i=0; i<6; i++) joint_position[i] = msg->data[i];
      for (int i=0; i<6; i++) joint_speed[i] = msg->data[i+6];
    }

    void SendInfo(const std_msgs::Float64MultiArray data){
    	info_pub.publish(data);
	    return;
    }
}; 


int main(int argc, char **argv)
{
  
  ros::init(argc, argv, "joint_controller_low");

  ros::NodeHandle n;

  ROS_INFO("Node Started");
  //--------------------------------
  GoalFollower my_follower;
  my_follower.info_pub = n.advertise<std_msgs::Float64MultiArray>("/info", 1);
  ros::Subscriber joint_status = n.subscribe("/joint_states_low", 1, &GoalFollower::change_states_msg, &my_follower);
  ros::Subscriber joint_goal = n.subscribe("/HighController/mpc_high_positions", 1, &GoalFollower::change_goal_msg, &my_follower);
  ros::Subscriber human_status = n.subscribe("/Obstacle/mpc_low_spheres", 1, &GoalFollower::change_obstacles_msg, &my_follower);
  ros::Publisher low_level_trajectory = n.advertise<nav_msgs::Path>("mpc_solver/low_level_trajectory", 1);
  ros::Publisher vis_pub = n.advertise<visualization_msgs::Marker>( "visualization_marker", 0 );
  ros::Publisher human_path = n.advertise<nav_msgs::Path>("human", 1);
  double smallest_dist;
  double local_val;
  // cposes of 7 test points:
  double ctp[21] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  // linear vels of 7 test points:
  double ctv[21] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  double ctv_linear[7] = {0,0,0,0,0,0,0};
  double min_dist[7] = {10000, 10000, 10000, 10000, 10000, 10000, 10000};
  ros::Rate loop_rate(20);

  clock_t begin = clock();
  double time_spent_mpc = 0;

  my_NMPC_solver myMpcSolver=my_NMPC_solver(5);

  while (ros::ok())
  {
    printf("max diff = %f\n",my_follower.from_high[31]);
    double current_joint_position[6];
    double current_human_position[56];
    double current_joint_vels[6];
    double cgoal[3];
    double goal[6];
    double tracking_goal[60];

    for (int i = 0; i < 6; ++i) current_joint_position[i] = my_follower.joint_position[i];
    for (int i = 0; i < 6; ++i) current_joint_vels[i] = my_follower.joint_speed[i];
    for (int i = 0; i < 6; ++i) goal[i] = my_follower.goal_queue[i];
    for (int i = 0; i < 3; ++i) cgoal[i] = my_follower.from_high[24+i];
    for (int i = 0; i < 60; ++i) tracking_goal[i] = my_follower.goal_queue[i];
    for (int i = 0; i < 56; ++i) current_human_position[i] = my_follower.human_sphere[i];

    ROS_INFO("Pose %f %f %f %f %f %f", current_joint_position[0], current_joint_position[1], current_joint_position[2], current_joint_position[3], current_joint_position[4], current_joint_position[5]);
  
    // Advance the goal
    for (int j=0; j<14; j++) for (int i=0; i<6; i++) my_follower.goal_queue[j*6+i] = my_follower.goal_queue[(j+1)*6+i]; 
  
    for (int i=0; i<6; i++) my_follower.goal_queue[14*6+i] = my_follower.goal_queue[13*6+i];

    //******************* get_min_dist **********************
	  smallest_dist = 10000;

    // for 7 points on UR5:
    min_dist[0] = 10000; min_dist[1] = 10000; min_dist[2] = 10000; 
    min_dist[3] = 10000; min_dist[4] = 10000; min_dist[5] = 10000; min_dist[6] = 10000; 

    double spheres_dist[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    Eigen::MatrixXf mat2 = get_cpose(my_follower.joint_position[0], my_follower.joint_position[1],
                                     my_follower.joint_position[2], my_follower.joint_position[3],
                                     my_follower.joint_position[4], my_follower.joint_position[5]);
   
    for (int j = 0; j < 7; j++) {
      Eigen::Vector3f w;
      w = mat2.col(j+1).transpose();
      ctp[j*3+0] = w[0];
      ctp[j*3+1] = w[1];
      ctp[j*3+2] = w[2];
      for (int k = 0; k < 14; k++) {
        Eigen::Vector3f p(my_follower.human_sphere[k*4+0], my_follower.human_sphere[k*4+1], my_follower.human_sphere[k*4+2]);
        local_val = dist_v(w, p) - my_follower.robot_spheres[j] - my_follower.human_sphere[k*4+3];
        if (local_val < min_dist[j]) {
          min_dist[j] = local_val;
          spheres_dist[j] = my_follower.robot_spheres[j] + my_follower.human_sphere[k*4+3];
        }
      }
	    if (smallest_dist > min_dist[j]) smallest_dist = min_dist[j];
	  }

    ROS_INFO("SD %f ", smallest_dist);
     
    double result[16]={0.0};
    double trajectory[66]={0.0};
    if (smallest_dist >= 0.00) {
      clock_t begin_mpc = clock();
      int status=myMpcSolver.solve_my_mpc(current_joint_position, goal, current_human_position, cgoal, tracking_goal, result, trajectory);   
      if (status > 0) {
            ROS_INFO("Destroying solver object");
            myMpcSolver.reset_solver();
            myMpcSolver=my_NMPC_solver(8);
            ROS_INFO("Solver recreated");
      }

      if (status==4) {
            for (int i=0; i<12; i++) result[i] = 0.0;
      }
      
      clock_t end_mpc = clock();
      double time_spent_mpc = (double)(end_mpc - begin_mpc) / CLOCKS_PER_SEC;

      double cgoal_trejectory[33];
        for (int i=0;i<11;i++) {
        Eigen::MatrixXf cgoal_trajectory_mat = get_cpose(trajectory[6*i+0], trajectory[6*i+1], 
		                trajectory[6*i+2], trajectory[6*i+3], trajectory[6*i+4], trajectory[6*i+5]);
        cgoal_trejectory[3*i+0] = cgoal_trajectory_mat.coeff(0, 7);
        cgoal_trejectory[3*i+1] = cgoal_trajectory_mat.coeff(1, 7);
        cgoal_trejectory[3*i+2] = cgoal_trajectory_mat.coeff(2, 7);
        //ROS_INFO("%f %f %f",cgoal_trejectory[3*i+0],cgoal_trejectory[3*i+1],cgoal_trejectory[3*i+2]);
        }
        // nav_msgs::Path path;
        // path.header.frame_id="/base_link";
        // geometry_msgs::PoseStamped pose;
        
        // pose.header.stamp = ros::Time::now();
        // pose.header.frame_id = "waypoint";
        // for (int i=0;i<11;i++) {
        //   pose.pose.position.x = cgoal_trejectory[3*i+0];
        //   pose.pose.position.y = cgoal_trejectory[3*i+1];
        //   pose.pose.position.z = cgoal_trejectory[3*i+2];
        //   path.poses.push_back(pose);
        // }
         
        // low_level_trajectory.publish(path);

        // nav_msgs::Path H;
        // H.header.frame_id="/base_link";
        // geometry_msgs::PoseStamped human;
        
        // human.header.stamp = ros::Time::now();
        // human.header.frame_id = "waypoint";
        // for (int i=0;i<14;i++) {
        //   human.pose.position.x = current_human_position[4*i+0];
        //   human.pose.position.y = current_human_position[4*i+1];
        //   human.pose.position.z = current_human_position[4*i+2];
        //   H.poses.push_back(human);
        // }
         
        // human_path.publish(H);

      } else for (int i=0; i<16; i++) result[i] = 0;

    //******************* get_min_velocity **********************
    double max_vell[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    Eigen::MatrixXf vell_mat = get_velocity(current_joint_position[0], current_joint_position[1], 
                                            current_joint_position[2], current_joint_position[3],
                                            current_joint_position[4], current_joint_position[5],
                                            result[0], result[1], result[2],
                                            result[3], result[4], result[5]);
    double max_linear_vell = 0;
    double temp_linear_vell = 0;
    
    for (int k=0; k<7; k++) {
      ctv[k*3+0] = vell_mat.coeff(k*3+0,0);
      ctv[k*3+1] = vell_mat.coeff(k*3+1,0);
      ctv[k*3+2] = vell_mat.coeff(k*3+2,0);
      temp_linear_vell = sqrt(vell_mat.coeff(k*3 + 0,0)*vell_mat.coeff(k*3 + 0,0) + vell_mat.coeff(k*3 + 1,0)*vell_mat.coeff(k*3 + 1,0) + vell_mat.coeff(k*3 + 2,0)*vell_mat.coeff(k*3 + 2,0));
      ctv_linear[k] = temp_linear_vell;
      max_vell[k] = temp_linear_vell;
      if (max_linear_vell < temp_linear_vell) max_linear_vell = temp_linear_vell;
    }

    //*********************** Apply control ********************************
	  double lin_vell_limit_arr[7] = {10, 10, 10, 10, 10, 10, 10};
    double lin_vell_scale = 10;

    double alpha[7] = {2.79, 1.95, 1, 0.8, 0.65, 0.45, 0.35};
    double d_bar= 0.15;
    double sqrt_temp_value = 0.00;
    for (int i=0; i<7; i++) {
      sqrt_temp_value = (min_dist[i]+spheres_dist[i])*(min_dist[i]+spheres_dist[i])-spheres_dist[i]*spheres_dist[i];
      if (sqrt_temp_value<0) lin_vell_limit_arr[i] = 0.00000000000000;
      else lin_vell_limit_arr[i] = alpha[i]*sqrt(sqrt_temp_value);
      double temp_scale = (lin_vell_limit_arr[i]/max_vell[i]);
      if (lin_vell_scale>temp_scale) lin_vell_scale = temp_scale;
    }

    if (lin_vell_scale<1.0) {
      for (int i = 0; i < 6; i++) result[i] = result[i]*lin_vell_scale;
    }
    ROS_INFO("Max lin vell %.3f, Vel Index %.3f,SD %.3f, ctp %.3f", max_linear_vell, lin_vell_scale, smallest_dist, ctp[17]);

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    
    // data sending
    std_msgs::Float64MultiArray whole_data;
    whole_data.data.clear();
    for (int i = 0; i < 6; i++) whole_data.data.push_back(current_joint_position[i]);
    for (int i = 0; i < 14; i++) for(int j = 0; j < 3; j++) whole_data.data.push_back(my_follower.human_sphere[i*4+j]);
    for (int i = 0; i < 6; i++) whole_data.data.push_back(result[i]);
    for (int i = 0; i < 21; i++) whole_data.data.push_back(ctp[i]);
    for (int i = 0; i < 6; i++) whole_data.data.push_back(my_follower.goal[i]);
    for (int i = 0; i < 6; i++) whole_data.data.push_back(my_follower.joint_speed[i]);
    for (int i = 0; i < 3; i++) whole_data.data.push_back(result[12+i]);
    for (int i = 0; i < 7; i++) whole_data.data.push_back(min_dist[i]);
    whole_data.data.push_back(smallest_dist);
    whole_data.data.push_back(lin_vell_scale);
    for (int i = 0; i < 30; i++) whole_data.data.push_back(my_follower.from_high[i]);
    for (int i = 0; i < 21; i++) whole_data.data.push_back(ctv[i]);
    for (int i = 0; i < 7; i++) whole_data.data.push_back(lin_vell_limit_arr[i]);
    whole_data.data.push_back(my_follower.human_sphere[56]);
    whole_data.data.push_back(time_spent);
    for (int i = 0; i < 7; i++) whole_data.data.push_back(ctv_linear[i]);
    whole_data.data.push_back(my_follower.from_high[31]);
    whole_data.data.push_back(my_follower.human_sphere[57]);
    whole_data.data.push_back(time_spent_mpc);
    my_follower.SendInfo(whole_data);
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}

