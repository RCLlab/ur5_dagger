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
using namespace std;
int ur_time=0;

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

// Introduce class to make safer goal change
class GoalFollower 
{ 
    // Access specifier 
    public: 
    // Data Members 
    ros::Publisher chatter_pub;
    ros::Publisher goal_state;
    double robot_spheres[7] = {0.15, 0.15, 0.15, 0.08, 0.08, 0.12, 0.1};
    double probabilities_predicted[2]={0};
    double human_sphere_predicted[560]={0}; 
    double goal[6] = {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000};
    double comand_vel[6] = {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000};
    double joint_position[6] = {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000};
    double joint_speed[6] = {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000};
  
    // Member Functions() 
    void change_goal(double new_goal[],int n) { 
       for (int i=0; i<n; i++) goal[i] = new_goal[i];
       ROS_INFO("Goal set to: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", 
	      goal[0], goal[1], goal[2], goal[3], goal[4], goal[5]); 
    }
    void change_obstacles_msg_predicted(const std_msgs::Float64MultiArray obstacle_data) { 
      for (int i=0; i<560; i++) human_sphere_predicted[i] = obstacle_data.data[i];
    }
    
    void change_goal_msg(const std_msgs::Float64MultiArray joint_pose_values) { 
       for (int i=0; i<6; i++) goal[i] = joint_pose_values.data[i];
       ROS_INFO("New goal set to: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", 
	   goal[0], goal[1], goal[2], goal[3], goal[4], goal[5]); 
    }

    void change_states_msg(const std_msgs::Float64MultiArray::ConstPtr& msg) { 
       for (int i=0; i<6; i++) joint_position[i] = msg->data[i];
    }

    void SendVelocity(const std_msgs::Float64MultiArray joint_vel_values){
    	chatter_pub.publish(joint_vel_values);
	    return;
    }
}; 

int main(int argc, char **argv)
{
  
  ros::init(argc, argv, "joint_controller_high");
  ros::NodeHandle n;
  ROS_INFO("Node Started");
  //--------------------------------
  GoalFollower my_follower;
  my_follower.chatter_pub = n.advertise<std_msgs::Float64MultiArray>("/HighController/mpc_high_positions", 1);
  my_follower.goal_state = n.advertise<std_msgs::String>("/HighController/goal_status", 1);
  ros::Publisher PauseHigh = n.advertise<std_msgs::Int32>("HighController/pause", 1);
  ROS_INFO("Goal default to: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f", 
	      my_follower.goal[0], my_follower.goal[1], my_follower.goal[2],
        my_follower.goal[3], my_follower.goal[4], my_follower.goal[5]);
  double read_goal[2][6] = { 3.0, -1.6, -1.7, -1.7, -1.7, 1.0,
                              0.0, -2.3, -1.1, -1.2, -1.2, 0.5};
                            //  1.7803887806843668, -2.909614979500528, 0.9936404219680592, 0.9536692218029104, 2.4373469898665125, -0.050777735063584295};
  int row_index_len = 2;
  double static_goal[6] = {read_goal[1][0], read_goal[1][1], read_goal[1][2], read_goal[1][3], read_goal[1][4], read_goal[1][5]};

  //------------------------------
  ros::Subscriber human_status = n.subscribe("/Obstacle/mpc_high_spheres", 1, &GoalFollower::change_obstacles_msg_predicted, &my_follower);
  ros::Subscriber joint_status = n.subscribe("/joint_states_high", 1, &GoalFollower::change_states_msg, &my_follower);
  ros::Publisher high_level_trajectory_main = n.advertise<nav_msgs::Path>("mpc_solver/high_level_trajectory_main", 1);
  ros::Publisher high_level_trajectory_second = n.advertise<nav_msgs::Path>("mpc_solver/high_level_trajectory_second", 1);
  
  my_NMPC_solver myMpcSolver=my_NMPC_solver(5);

  std_msgs::Float64MultiArray joint_vel_values;
  double cgoal[3];

  // Big loop
  double loop_duration = 149.366667; // no pauses
  double start_motion_time = 4.00;
  double stop_human_time = (loop_duration*2);
  for (int big_loop_iteration=0;big_loop_iteration<3;big_loop_iteration++) {
    start_motion_time = big_loop_iteration*0.5 + 4.0;
    stop_human_time = big_loop_iteration*(loop_duration*2) + (loop_duration*2);
    ROS_INFO("Time: %.3f, %.3f", start_motion_time, stop_human_time);

    int row_index = 0;
    double loop_start_time = 0;

    ros::Rate goto_loop(20);
    while (ur_time < 100){
      joint_vel_values.data.clear();
            for (int i = 0; i < 6; i++) joint_vel_values.data.push_back(0.0);
            for (int i = 0; i < 6; i++) joint_vel_values.data.push_back(0.0);
            for (int i = 0; i < 6; i++) joint_vel_values.data.push_back(static_goal[i]);
      
      Eigen::MatrixXf cgoal_mat = get_cpose(static_goal[0], static_goal[1], 
          static_goal[2], static_goal[3], static_goal[4], 
          static_goal[5]);
      cgoal[0] = cgoal_mat.coeff(0, 7);
      cgoal[1] = cgoal_mat.coeff(1, 7);
      cgoal[2] = cgoal_mat.coeff(2, 7);
      for (int i = 0; i < 3; i++) joint_vel_values.data.push_back(cgoal[i]);
      my_follower.SendVelocity(joint_vel_values);
      ur_time++;
      ros::spinOnce();
      goto_loop.sleep();
    };

	  int task = -1;
    int task_started = 0;
	  ros::Rate loop_rate(2);
    while (ros::ok())
	  {
	    if (row_index==0) {
        if (task_started == 0) {
			    task = task + 1;
		    }
        task_started = 1;
      }
      double current_joint_position[6];
      double current_human_position_predicted[560]={0};
      double current_joint_goal[6];
      for (int i = 0; i < 6; ++i) current_joint_position[ i ] = my_follower.joint_position[ i ];
      for (int i = 0; i < 6; ++i) current_joint_goal[ i ] = read_goal[row_index][i];
      for (int i = 0; i < 560; ++i) current_human_position_predicted[ i ] = my_follower.human_sphere_predicted[ i ];

      Eigen::MatrixXf cgoal_mat = get_cpose(read_goal[row_index][0], read_goal[row_index][1], 
		       read_goal[row_index][2], read_goal[row_index][3], read_goal[row_index][4], 
		       read_goal[row_index][5]);
      cgoal[0] = cgoal_mat.coeff(0, 7);
      cgoal[1] = cgoal_mat.coeff(1, 7);
      cgoal[2] = cgoal_mat.coeff(2, 7);

      ROS_INFO("Creating solver");
      ROS_INFO("Pose %f %f %f %f %f %f", current_joint_position[0], current_joint_position[1], current_joint_position[2], current_joint_position[3], current_joint_position[4], current_joint_position[5]);
      ROS_INFO("Goal %f %f %f %f %f %f", current_joint_goal[0], current_joint_goal[1], current_joint_goal[2], current_joint_goal[3], current_joint_goal[4], current_joint_goal[5]);
      double result[16]={0.0};
      double trajectory_first[66]={0.0};
      double trajectory_second[66]={0.0};
      int status=myMpcSolver.solve_my_mpc(current_joint_position, current_human_position_predicted, current_joint_goal, cgoal, result, trajectory_first);
      if (status > 0) {
        ROS_INFO("Destroying solver object");
        myMpcSolver.reset_solver();
        myMpcSolver=my_NMPC_solver(8);
        ROS_INFO("Solver recreated");
      }

      if (status==4) {
        for (int i=0; i<14; i++) result[i] = 0.0;
      }
      ROS_INFO("KKT %f; Status %i",result[14], status);

      double cgoal_trejectory_first[33]={0.0};
      double cgoal_trejectory_second[33]={0.0};
      for (int i=0;i<11;i++) {
        Eigen::MatrixXf cgoal_trajectory_mat1 = get_cpose(trajectory_first[6*i+0], trajectory_first[6*i+1], 
            trajectory_first[6*i+2], trajectory_first[6*i+3], trajectory_first[6*i+4], trajectory_first[6*i+5]);
        cgoal_trejectory_first[3*i+0] = cgoal_trajectory_mat1.coeff(0, 7);
        cgoal_trejectory_first[3*i+1] = cgoal_trajectory_mat1.coeff(1, 7);
        cgoal_trejectory_first[3*i+2] = cgoal_trajectory_mat1.coeff(2, 7);
      }

      // Check if arrived
      std_msgs::String goal_state_msg;
      std::stringstream ss;
      float max_diff = 0;
      for (int i = 0; i < 6; i++) {
          if (abs(current_joint_position[i] - current_joint_goal[i]) > max_diff) {
              max_diff = abs(current_joint_position[i] - current_joint_goal[i]); 
          }
      }
      ROS_INFO("max_diff %f",max_diff);
      if (max_diff < 0.02) {
        row_index = (row_index+1)%row_index_len;
      }
      // else ss << "Following";
      goal_state_msg.data = ss.str();
      my_follower.goal_state.publish(goal_state_msg);
      //******************* get_min_dist **********************
	    float local_val = 10000;
	    double smallest_dist = 10000;
	    double min_dist[] = {10000, 10000, 10000, 10000, 10000, 10000, 10000};
      Eigen::MatrixXf mat2 = get_cpose(my_follower.joint_position[0], my_follower.joint_position[1], 
            my_follower.joint_position[2], my_follower.joint_position[3], my_follower.joint_position[4], 
            my_follower.joint_position[5]);
	    for (int j = 0; j<7; j++) {
        Eigen::Vector3f w;
        w = mat2.col(j+1).transpose();
        for (int i = 0; i < 14; i++) {
          Eigen::Vector3f p(my_follower.human_sphere_predicted[i*4+0],my_follower.human_sphere_predicted[i*4+1],my_follower.human_sphere_predicted[i*4+2]);
          local_val = dist_v(w, p) - my_follower.robot_spheres[j] - my_follower.human_sphere_predicted[i*4+3];
          if (min_dist[j] > local_val) min_dist[j] = local_val;
        }
        if (smallest_dist > min_dist[j]) smallest_dist = min_dist[j];
	    }
      // prepare to send commands
      joint_vel_values.data.clear();
      for (int i = 0; i < 12; i++) joint_vel_values.data.push_back(result[i]);
      for (int i = 0; i < 6; i++) joint_vel_values.data.push_back(current_joint_position[i]);
      for (int i = 0; i < 6; i++) joint_vel_values.data.push_back(current_joint_goal[i]);
      for (int i = 0; i < 3; i++) joint_vel_values.data.push_back(cgoal[i]);
      for (int i = 0; i < 4; i++) joint_vel_values.data.push_back(result[12+i]);
      joint_vel_values.data.push_back(max_diff);
      my_follower.SendVelocity(joint_vel_values);
    ros::spinOnce();
    loop_rate.sleep();
	  }
  }

  return 0;
}

