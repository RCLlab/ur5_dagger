git clone https://github.com/catkin/catkin_simple.git
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
pip install rospkg
pip install pandas
pip install pyyaml


install python 3.8.10
then in venv install
pip install pytorch
pip install numpy
pip install scipy
pip install pyyaml
pip install rospkg


## This is UR5 controllers with acados

## BASHRC check
1. gedit ~/.bashrc
2. edit path
3. source ~/.bashrc

## Testing with Real robot
1. robot ip parameters: ip: 192.168.1.2/ mask:255.255.255.0/ gateway:192.168.1.1
2. computer ip parameters: ip: 192.168.1.1 / mask:255.255.255.0/ gateway: 192.168.1.1

## Install packages
1. pip install pyyaml
2. pip install rospkg
3. pip install pandas
4. pip install pytorch

## URSIM install
install `ursim-5.9.4.10321232`

## DIL training (1.12.22)
1. ./start-ursim.sh
2. roslaunch ur_modern_driver ur5_bringup.launch robot_ip:=127.0.0.1
3. rosrun human_vrep human_sim
4. rosrun human_vrep human_spheres.py
5. rosrun mpc_low move_low_node
6. rosrun mpc_high move_high_node
7. source ./venv/bin/active
cd workspaces/ur5_dagger/src/dil_train
./action_send.py
	


