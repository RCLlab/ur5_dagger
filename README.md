# UR5 Controllers with Acados

## Installation and Setup

### Step 1: Clone the Repository

1. Clone the `catkin_simple` repository:
   ```bash
   git clone https://github.com/catkin/catkin_simple.git

    Build the workspace:

    catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3

Step 2: Install Python Dependencies
Install necessary Python packages:
      pip install pandas

Install Python 3.8.10 if not already installed.
Set up a virtual environment (venv) and install additional packages:

    pip install pytorch
    pip install numpy
    pip install scipy
    pip install pyyaml
    pip install rospkg

BASHRC Configuration

Open the ~/.bashrc file:

    gedit ~/.bashrc
Edit the PATH as necessary.

Apply the changes:

    source ~/.bashrc
    
Testing with Real Robot

Robot IP Configuration:

IP: 192.168.1.2

Subnet Mask: 255.255.255.0

Gateway: 192.168.1.1

Computer IP Configuration:

IP: 192.168.1.1

Subnet Mask: 255.255.255.0

Gateway: 192.168.1.1

URSIM Installation

Install URSIM version 5.9.4.10321232.


DIL Training 

Start URSIM:
    ./start-ursim.sh

    
Launch UR5 bringup:


   roslaunch ur_modern_driver ur5_bringup.launch robot_ip:=127.0.0.1
   

Run human simulation:


   rosrun human_vrep human_sim


Run human spheres:


   rosrun human_vrep human_spheres.py
   

Run the low-level MPC node:


   rosrun mpc_low move_low_node
   

Run the high-level MPC node:


   rosrun mpc_high move_high_node
   

Activate your virtual environment:


   source ./venv/bin/activate


Navigate to the workspace and run the training script:


   cd workspaces/ur5_dagger/src/dil_train
   
   ./action_send.py




MPC Test

Start URSIM:


   ./start-ursim.sh


Launch UR5 bringup:


   roslaunch ur_modern_driver ur5_bringup.launch robot_ip:=127.0.0.1
   

Run human test script:


   rosrun human_vrep test_human.py
   

Run human simulation:


   rosrun human_vrep human_sim
   

Run the low-level MPC node:


   rosrun mpc_low mpc_low_node
   

Run the high-level MPC node:


   rosrun mpc_high mpc_high_node
   

Run the low-level MPC one-way script:


    rosrun mpc_low one_way.py
    


Dagger Test 


Start URSIM:


   ./start-ursim.sh
   

Launch UR5 bringup:


   roslaunch ur_modern_driver ur5_bringup.launch robot_ip:=127.0.0.1
   

Run human test script:


   rosrun human_vrep test_human
   

Run human spheres:


   rosrun human_vrep human_spheres.py
   

Run the low-level MPC node:


   rosrun mpc_low move_low_node
   

Run the high-level MPC node:


   rosrun mpc_high move_high_node
   

Run the low-level MPC one-way script:


    rosrun mpc_low_node one_way.py

    

E-Dagger Training (05.06.23)


Open the URSIM UR5 app.

Launch UR5 bringup:


   roslaunch ur_modern_driver ur5_bringup.launch robot_ip:=127.0.0.1
   

Run human training script:


   rosrun human_vrep train_human.py
   

Run human simulation:


   rosrun human_vrep human_sim
   

Run the low-level MPC node:


   rosrun mpc_low mpc_low_node
   

Run the high-level MPC node:


   rosrun mpc_high mpc_high_node
   

Activate your DNN environment:


   source ./DDNV/bin/activate
   

Navigate to the workspace and run the E-Dagger training script:


    cd workspaces/ur5_dagger/src/dil_train
    
    python ./e_dagger_train.py


E-Dagger Test


Navigate to the workspace:


   cd workspaces/ur5_dagger/src/dil_train
   

Run human test script:


   ./test_human.py
   

Run the control script:


   ./control.py

Run the E-Dagger test script:

   ./e_dagger_test.py
