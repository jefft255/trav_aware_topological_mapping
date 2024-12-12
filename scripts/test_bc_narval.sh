# Script to test the behavior cloning model on the simulator
# $1 is path to run output dir with weight file, tensorboards logs and hydra config
# $2 is the run number, because we run multiple test runs due to high variance

module load apptainer-suid

echo "Killing previous simulators"
kill -9 $(ps -aux | grep "ASE_MWConiferForest" | awk '{print $2}')
kill -9 $(ps -aux | grep "roscore" | awk '{print $2}')
kill -9 $(ps -aux | grep "bc_node.py" | awk '{print $2}')
kill -9 $(ps -aux | grep "apptainer" | awk '{print $2}')
kill -9 $(ps -aux | grep "Apptainer" | awk '{print $2}')

if [ "$3" = "only_front" ];
then
    CONFIG_FILE=/home/jftrem/projects/def-dpmeger/jftrem/topological_mapping/config/rc_car_1cam_airsim_settings.json
else
    CONFIG_FILE=/home/jftrem/projects/def-dpmeger/jftrem/topological_mapping/config/rc_car_airsim_settings.json
fi

# Only copy files and build in the first run!
if [[ "$2" == "0" ]];
then
    echo "Copying stuff..."
    # /bin/cp is to remove the alias to cp -i. This means no confirmation to overwrite is needed.
    /bin/cp -r /home/jftrem/projects/def-dpmeger/jftrem/stefan_game_engine.zip ${SLURM_TMPDIR}
    unzip -q -o ${SLURM_TMPDIR}/stefan_game_engine.zip -d ${SLURM_TMPDIR}
    /bin/cp /home/jftrem/jf.sif ${SLURM_TMPDIR}
    /bin/cp -r /home/jftrem/projects/def-dpmeger/jftrem/AirSim.zip ${SLURM_TMPDIR}
    unzip -q -o ${SLURM_TMPDIR}/AirSim.zip -d ${SLURM_TMPDIR}
    /bin/cp -r /home/jftrem/projects/def-dpmeger/jftrem/topological_mapping ${SLURM_TMPDIR}/AirSim/ros/src/

    # For some reason /tmp can't handle all the crap from compiling
    mkdir ${SLURM_TMPDIR}/tmp

    echo "Building catkin workspace"
    # Have to mount /tmp because it's full of crap from compiling, and AllianceCan FS is too small
    apptainer exec -C -B ${SLURM_TMPDIR} -B ${SLURM_TMPDIR}/tmp:/tmp ${SLURM_TMPDIR}/jf.sif \
        bash -c "cd ${SLURM_TMPDIR}/AirSim/ros && \
        source /opt/ros/noetic/setup.bash && \
        rm -r build && \
        rm -r devel && \
        catkin_make  -DCMAKE_TMPDIR=${SLURM_TMPDIR}/tmp 1> /dev/null 2> /dev/null"
fi


echo "Starting simulator"
# This seems to crash randomly. Create a loop that keeps trying to start the simulator
while true
do
    bash ${SLURM_TMPDIR}/stefan_game_engine/ASE_MWConiferForest.sh \
    -nosound -windowed -vulkan -RenderOffscreen -windowed -ResX=320 -ResY=240 --settings \
    $CONFIG_FILE 1> /dev/null 2> /dev/null &
    sleep 10
    isalive=$(ps -aux | grep "ASE_MWConiferForest" | wc -l)
    if [ "$isalive" -gt 1 ]
    then
        echo "Simulator successfully started"
        break
    else
        echo "Simulator crashed, restarting"
    fi
done


echo "Launching roscore"
apptainer exec -C --nv -B ${SLURM_TMPDIR}  ${SLURM_TMPDIR}/jf.sif \
    bash -c "source ${SLURM_TMPDIR}/AirSim/ros/devel/setup.bash && \
    roscore" &
sleep 5

echo "Launching rosbag"
apptainer exec -C --nv -B ${SLURM_TMPDIR}  ${SLURM_TMPDIR}/jf.sif \
    bash -c "source ${SLURM_TMPDIR}/AirSim/ros/devel/setup.bash && \
    rosbag record -a -O ${SLURM_TMPDIR}/data.bag " &


echo "Launching AirSim ros wrapper"
apptainer exec -C --nv -B ${SLURM_TMPDIR}  ${SLURM_TMPDIR}/jf.sif \
    bash -c "source ${SLURM_TMPDIR}/AirSim/ros/devel/setup.bash && \
    roslaunch topological_mapping airsim_node.launch 1> /dev/null 2> /dev/null" &


echo "Starting BC node"
apptainer exec -C --nv -B ${SLURM_TMPDIR} -B $1 -B /home/jftrem/projects/def-dpmeger/jftrem -B /cvmfs  ${SLURM_TMPDIR}/jf.sif \
    bash -c "source ${SLURM_TMPDIR}/AirSim/ros/devel/setup.bash && \
    python3 \
    ${SLURM_TMPDIR}/AirSim/ros/src/topological_mapping/scripts/bc_node.py $1" &

echo "Starting test script"
mkdir $1/test_run_$2
apptainer exec -C --nv -B ${SLURM_TMPDIR} -B $1 -B /home/jftrem/projects/def-dpmeger/jftrem -B /cvmfs  ${SLURM_TMPDIR}/jf.sif \
    bash -c "source ${SLURM_TMPDIR}/AirSim/ros/devel/setup.bash && \
    python3 \
    ${SLURM_TMPDIR}/AirSim/ros/src/topological_mapping/scripts/test_bc.py > $1/test_run_$2/test_bc_results.csv"

kill -15 $(ps -aux | grep "rosbag" | awk '{print $2}')
sleep 2
cp ${SLURM_TMPDIR}/data.bag $1/test_run_$2/
