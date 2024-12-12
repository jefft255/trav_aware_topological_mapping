#!/bin/zsh

# Set the directory where you want to search
# directory="/media/jft/diskstation/bagfiles"
directory="/media/jft/diskstation/bagfiles/gault_sept11"
output_directory="/media/jft/diskstation/bc_data_test"

mkdir -p $output_directory

# Set the extension you're interested in
extension=".bag"

# conda activate ros_env
files=$(find "$directory" -type f -name "*$extension")

# Loop through each file
shopt -s globstar
for file in $directory/**/*.bag; do
    # Use basename to extract the filename without path
    basenamef=$(basename "$file")

    # Use parameter expansion to remove the extension
    filename_without_extension="${basenamef%.*}"
    if [[ $filename_without_extension == *"data_20"* ]]; then
        echo "Skipping '$file'"
    else
        kill -9 $(ps -aux | grep "roscore" | awk '{print $2}')
        kill -9 $(ps -aux | grep "rosbag" | awk '{print $2}')
        kill -9 $(ps -aux | grep "roslaunch" | awk '{print $2}')
        kill -9 $(ps -aux | grep "navsat" | awk '{print $2}')
        kill -9 $(ps -aux | grep "localization" | awk '{print $2}')
        kill -9 $(ps -aux | grep "imu_tf" | awk '{print $2}')
        kill -9 $(ps -aux | grep "gps_tf" | awk '{print $2}')

        echo "Processing '$file'"
        # roslaunch the localization stack
        # and capture its PID
        rosparam set /use_sim_time true
        roslaunch husky_localization husky_localization.launch &
        sleep 5
        pid_loc=$(ps -aux | grep "robot_localization" | awk '{print $2}')

        # Run the BC dataset creation node, and save to output_dir with the filename of the bag
        python /home/adaptation/jft/catkin_ws/src/topological_mapping/scripts/bc_dataset_creation_node.py "$output_directory/$filename_without_extension.pkl" &
        sleep 5

        # play the rosbag file
        rosbag play --clock -r 0.5 $file &
        pid_bag=$!
        playing=true
        sleep 5

        while "$playing"; do
            # check if the rosbag is still running
            if ! kill -0 $pid_bag; then
                # if not, break the loop
                echo "Rosbag finished playing, next one"
                playing=false
            fi
            sleep 5
        done
        # Sleep for a while to make sure the data processing script has saved to disk
        sleep 40
    fi
done

kill -9 $(ps -aux | grep "roscore" | awk '{print $2}')
kill -9 $(ps -aux | grep "rosbag" | awk '{print $2}')
kill -9 $(ps -aux | grep "roslaunch" | awk '{print $2}')
kill -9 $(ps -aux | grep "navsat" | awk '{print $2}')
kill -9 $(ps -aux | grep "localization" | awk '{print $2}')
kill -9 $(ps -aux | grep "imu_tf" | awk '{print $2}')
kill -9 $(ps -aux | grep "gps_tf" | awk '{print $2}')
