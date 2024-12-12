#!/usr/bin/zsh

# This kill the background process (the simulator) when the script is killed
echo "Killing previous simulators"
kill -9 $(ps -aux | grep "ASE_MWConiferForest" | awk '{print $2}')
kill -9 $(ps -aux | grep "roscore" | awk '{print $2}')
kill -9 $(ps -aux | grep "bc_node.py" | awk '{print $2}')
kill -9 $(ps -aux | grep "apptainer" | awk '{print $2}')
kill -9 $(ps -aux | grep "Apptainer" | awk '{print $2}')

if [ "$3" = "only_front" ];
then
    CONFIG_FILE=~/catkin_ws/src/topological_mapping/config/rc_car_1cam_airsim_settings.json
else
    CONFIG_FILE=~/catkin_ws/src/topological_mapping/config/rc_car_airsim_settings.json
fi

zsh /localdata/jft/bin_unreal/LinuxNoEditor/ASE_MWConiferForest.sh -windowed -ResX=1600 -ResY=900 --settings $CONFIG_FILE &
sleep 10
source /localdata/jft/Unreal\ Projects/AirSim/ros/devel/setup.zsh
roslaunch topological_mapping airsim_node.launch 1> /dev/null 2> /dev/null &
sleep 5
# rosbag record -a -O $1/test_run_$2/data.bag &
sleep 5
python /home/adaptation/jft/catkin_ws/src/topological_mapping/scripts/bc_node.py $1 &
sleep 5
# Run test script which will print out a csv with the results
mkdir $1/test_run_$2
python /home/adaptation/jft/catkin_ws/src/topological_mapping/scripts/test_bc.py > $1/test_run_$2/test_bc_results.csv
for pid in $(ps -ef | grep "ASE_" | awk '{print $2}');
do
    kill -9 $pid;
done
kill -15 $(ps -aux | grep "rosbag" | awk '{print $2}')