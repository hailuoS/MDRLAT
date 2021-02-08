# MDRLAT
Training MDRLAT in Gazebo
Launch the Gazebo world:
roslaunch turtlebot_gazebo turtlebot_world.launch world_file:=/PATH TO/MDRLAT/world/Training.world
Start training
python MDRLAT_train.py
Start testing in virtual environment
python MDRLAT_test.py
