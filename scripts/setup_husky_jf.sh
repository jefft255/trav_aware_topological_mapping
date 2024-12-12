bash ~/catkin_ws/src/topological_mapping/scripts/scale_screen.sh
rosnode kill /ekf_localization
rviz -d ~/catkin_ws/src/topological_mapping/config/topological_mapping.rviz &
