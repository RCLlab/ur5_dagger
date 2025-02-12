## Safely Imitating Predictive Control Policies for Real-Time Human-Aware Manipulator Motion Planning: A Dataset Aggregation Approach

This repository contains the source code for the experiments and results discussed in the paper titled "Safely Imitating Predictive Control Policies for Real-Time Human-Aware Manipulator Motion Planning: A Dataset Aggregation Approach". The code demonstrates the implementation of the algorithms described and provides tools to replicate our findings.

**Authors**: Aigerim Nurbayeva and Matteo Rubagotti

You can read the full paper here: [Link to Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10819386)

## Download the workspace
```bash
git clone --recurse-submodules https://github.com/RCLlab/ur5_dagger
   ```
Build the ROS environment with the following command:
```bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

## Dependencies
To run the code, you need the following:
- pandas
- python3.8

Install pandas using the following command:
```bash
pip install pandas
```

Create virtual environment using the following command:
```bash
python3.8 - m venv venv
```

Install packages inside of the virtual environment using the following commands:
```bash
pip install pytorch
pip install numpy
pip install scipy
pip install pyyaml
pip install rospkg
```

## Implementation and Testing Environment
The algorithms are implemented and tested using ROS on Ubuntu 20.04 with the real robot. Additionally, before testing on the real robot, they were simulated using the URSim simulator 'ursim-5.9.4.10321232'. For installation instructions for URSim, please follow the guidelines available [here](https://www.universal-robots.com/download/?query=).


Ensure the ROS environment is sourced correctly by adding the following lines to your `.bashrc` file:
```bash
gedit ~/.bashrc
# Add the following line at the end of the file:
# source /path/to/your/catkin_workspace/devel/setup.bash
source ~/.bashrc
```

## Training the Algorithms
To test the algorithms on the robot, first configure the TCP/IP parameters for both the robot and your computer

Execute the following steps in different terminals:
**DIL Training**
1. **Launch Robot Drivers:**
   ```bash
   roslaunch ur_modern_driver ur5_bringup.launch robot_ip:=192.168.1.2
   ```
2. **Test Human Movement Simulation:**
   ```bash
   rosrun human_vrep train_human.py
   ```
3. **Send Human Movement Data to Controller:**
   ```bash
   rosrun human_vrep human_sim
   ```
4. **Run Low-Level Controller:**
   ```bash
   rosrun mpc_low mpc_low_node
   ```
5. **Run High level Controller:**
   ```bash
   rosrun mpc_high mpc_high_node
   ```
6. **Activate your virtual environment and navigate to the workspace::**
   ```bash
   source ./venv/bin/activate
   cd workspaces/ur5_dagger/src/dil_train
   ```
7.1. **Running Models Without Encoder (O-DNN, ODA-DNN, DA-DNN)**

To run models without an encoder, follow these steps:
Open the file dagger_train.py located in the dil_train directory.
Specify the following parameters in the script:
**Direction:**
```python
direction = 'AB/'  # or 'BA/'
```
**Controller:**
```python
controller = 'O-DNN/'  # or 'ODA-DNN' or 'DA-DNN'
```

**Safety Mode:**
```python
safety = 'NotSafe/'  # or 'Safe/'
```

Run the script using the command:
   ```bash
   ./dagger_train.py 
   ```
**E-dagger Training**
1. **Launch Robot Drivers:**
   ```bash
   roslaunch ur_modern_driver ur5_bringup.launch robot_ip:=192.168.1.2
   ```
2. **Test Human Movement Simulation:**
   ```bash
   rosrun human_vrep train_human.py
   ```
3. **Send Human Movement Data to Controller:**
   ```bash
   rosrun human_vrep human_sim
   ```
4. **Run Low-Level Controller:**
   ```bash
   rosrun mpc_low mpc_low_node
   ```
5. **Run High level Controller:**
   ```bash
   rosrun mpc_high mpc_high_node
   ```
6. **Activate your virtual environment, Navigate to the workspace and run the training script::**
   ```bash
   source ./venv/bin/activate
   cd workspaces/ur5_dagger/src/dil_train
   python ./e_dagger_train.py
   ```

## Testing the Algorithms
**E-dagger Testing**
1. **Launch Robot Drivers:**
   ```bash
   roslaunch ur_modern_driver ur5_bringup.launch robot_ip:=192.168.1.2
   ```
2. **Test Human Movement Simulation:**
   ```bash
   cd /ur5_dagger/src/dil_train
   ./test_human.py
   ```
3. **Send Human Movement Data to Controller:**
   ```bash
   cd /ur5_dagger/src/dil_train
   ./control.py
   ```
4. **Activate your virtual environment and run the script::**
   ```bash
   source ./venv/bin/activate
   python ./e_dagger_test.py
   ```

## Testing the Algorithms
**MPC Testing**
1. **Launch Robot Drivers:**
   ```bash
   roslaunch ur_modern_driver ur5_bringup.launch robot_ip:=192.168.1.2
   ```
2. **Test Human Movement Simulation:**
   ```bash
   rosrun human_vrep test_human.py
   ```
3. **Send Human Movement Data to Controller:**
   ```bash
   rosrun human_vrep human_sim
   ```
4. **Run Low-Level Controller:**
   ```bash
   rosrun mpc_low mpc_low_node
   ```
5. **Run High level Controller:**
   ```bash
   rosrun mpc_high mpc_high_node
   ```
6. **Run the MPC one-way script:::**
   ```bash
   rosrun mpc_low one_way.py
   ```
   
## Additional Resources
The human dataset used for experiments, known as AnDyDataset, can be accessed [here](https://andydataset.loria.fr/).

## Contact
For questions and feedback, please reach out to Aigerim Nurbayeva at aigerim.nurbayeva@nu.edu.kz
```
