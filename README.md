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
**DAgger training (DA-DNN)**
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
To select a direction in the **high_solver.cpp** file located on the mpc_high page, specify the row index, it can be 0 (AtoB) or 1(BtoA).

5. **Run High level Controller:**
   ```bash
   rosrun mpc_high mpc_high_node
   ```
6. **Activate your virtual environment and navigate to the workspace::**
   ```bash
   source ./venv/bin/activate
   cd workspaces/ur5_dagger/src/dil_train
   ```
7. **Run the script using the command:**
   ```bash
   ./dagger_train.py 
   ```
In order to train O-DNN, and ODA-DNN you need first collect dataset by running MPC controller. If you have dataset, you can run script on your venv:
   ```bash
   ./dnn_train.py 
   ```

**E-DAgger (E-DA-DNN) training is similar to DAgger training except step 7**
7. **Run the script using the command:**
   ```bash
   ./e_dagger_train.py 
   ```
In order to train O-E-DNN, and ODA-E-DNN you can use the same dataset that you used for O-DNN and ODA-DNN:
   ```bash
   ./e_dnn_train.py 
   ```

## Testing the Algorithms
**Testing the Models With Encoder (O-E-DNN, ODA-E-DNN, DA-E-DNN, O-SE-DNN, ODA-SE-DNN, DA-SE-DNN)**
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
Open the file **e_dagger_test.py** located in the 'dil_train' directory.
Specify the following parameters in the script:

**Direction:**
```python
direction = 'AB/'  # or 'BA/'
```
**Controller:**
```python
controller = 'E-O-DNN/'  # or 'E-ODA-DNN' or 'E-DA-DNN'
```
**Safety Mode:**
```python
safety = 'NotSafe/'  # or 'Safe/'
```

4. **Activate your virtual environment and run the script::**
   ```bash
   source ./venv/bin/activate
   python ./e_dagger_test.py
   ```

**Testing the Models Without Encoder (O-DNN, ODA-DNN, DA-DNN, O-S-DNN, ODA-S-DNN, DA-S-DNN)**
Repeat steps 1-3

Open the file **dagger_test.py** located in the 'dil_train' directory.
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

4. **Activate your virtual environment and run the script::**
   ```bash
   source ./venv/bin/activate
   python ./dagger_test.py
   ```
   
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

Open the file **one_way.py** located in the 'mpc_low' directory.
Specify the following parameters in the script:

**Direction:**
```python
direction = 'AB/'  # or 'BA/'
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
