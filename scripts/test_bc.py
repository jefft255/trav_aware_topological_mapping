#!/usr/bin/env python3
from typing import Any
from typing_extensions import Literal
import numpy as np
import pickle

import rospy

from topological_mapping.msg import GoToGoalBCAction, GoToGoalBCGoal
from actionlib import SimpleActionClient
from actionlib import GoalStatus
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_matrix


last_odom = None


def create_goal_from_pose() -> GoToGoalBCGoal:
    global last_odom
    # Goal is assumed to be in odom frame
    # Or actually in whatever frame /odometry/filtered is
    distance = np.random.uniform(2.0, 5.0)
    theta = np.random.uniform(-np.pi / 2, np.pi / 2)
    rot_matrix = quaternion_matrix(
        [
            last_odom.pose.pose.orientation.x,
            last_odom.pose.pose.orientation.y,
            last_odom.pose.pose.orientation.z,
            last_odom.pose.pose.orientation.w,
        ]
    )
    x_axis_2d = rot_matrix[:2, 0] / np.linalg.norm(rot_matrix[:2, 0])
    goal_vector_x = x_axis_2d[0] * np.cos(theta) - x_axis_2d[1] * np.sin(theta)
    goal_vector_y = x_axis_2d[0] * np.sin(theta) + x_axis_2d[1] * np.cos(theta)
    rospy.logwarn(f"{x_axis_2d=}")
    rospy.logwarn(f"{goal_vector_x=}, {goal_vector_y=}")
    goal = GoToGoalBCGoal()
    goal.goal.x = last_odom.pose.pose.position.x + distance * goal_vector_x
    goal.goal.y = last_odom.pose.pose.position.y + distance * goal_vector_y
    goal.goal.z = last_odom.pose.pose.position.z  # Doesn't really matter
    return goal


def odom_callback(msg: Odometry):
    global last_odom
    last_odom = msg


if __name__ == "__main__":
    rospy.init_node("bc_tester")
    rate = rospy.Rate(10)
    action_client = SimpleActionClient("bc", GoToGoalBCAction)
    odom_subscriber = rospy.Subscriber("/odometry/filtered", Odometry, odom_callback)
    n_goals = 15
    results = []
    i = 0

    while not rospy.is_shutdown():
        action_client.wait_for_server()
        if last_odom is not None:
            goal = create_goal_from_pose()
            goal_state = action_client.send_goal_and_wait(goal)
            i += 1
            # print(f"{i=}")
            # print(goal_state)
            results.append(goal_state)
            if i == n_goals:
                break
        rate.sleep()
    n_fails = sum([r != 3 for r in results])
    n_success = sum([r == 3 for r in results])
    print(f"Success rate,{float(n_success) / float(n_goals)}")
    print(f"N success,{n_success}")
