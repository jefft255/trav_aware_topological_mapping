#!/usr/bin/env python
import rospy
import actionlib
import sys
from pathlib import Path
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Twist
from threading import Lock, Thread
from typing import Optional
from topological_mapping.msg import (
    GoToGoalBCAction,
    GoToGoalBCActionGoal,
    GoToGoalBCActionFeedback,
    GoToGoalBCActionResult,
    GoToGoalBCResult,
)
import cv2
from cv_bridge import CvBridge

from tf.transformations import quaternion_matrix

import numpy.typing as npt
import numpy as np


def quaternion_matrix_custom(quat):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array(
        [[r00, r01, r02, 0], [r10, r11, r12, 0], [r20, r21, r22, 0], [0, 0, 0, 1]]
    )

    return rot_matrix


class CameraAccumulator:
    def __init__(self, hist_size, only_front) -> None:
        self.bridge = CvBridge()
        self.hist_size = hist_size
        self.only_front = only_front
        self.last_cmd: Twist = Twist()
        self.last_command: Optional[npt.NDArray[np.float32]] = None
        self.last_position: Optional[npt.NDArray[np.float64]] = None
        self.last_position_t = Optional[rospy.Time]
        self.first_position: Optional[npt.NDArray[np.float64]] = None
        self.last_orientation: Optional[npt.NDArray[np.float64]] = None
        self.prev_pos_for_vel: Optional[npt.NDArray[np.float64]] = None
        self.odom_subscriber = rospy.Subscriber(
            "/odometry/filtered", Odometry, self.odom_callback
        )
        self.cmd_subscriber = rospy.Subscriber(
            "/husky_velocity_controller/cmd_vel", Twist, self.cmd_callback
        )
        self.pinned_pose = None
        self.time_resolution = rospy.Duration(0.2)
        self.last_save_t = None
        self.lock = Lock()

    def cmd_callback(self, cmd_msg: Twist):
        self.last_cmd = cmd_msg

    def odom_callback(self, msg: Odometry):
        self.last_position = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]
        )
        self.last_position_t = msg.header.stamp

        self.last_orientation = quaternion_matrix(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ]
        )

        if self.first_position is None:
            self.first_position = np.array(
                [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ]
            )
            self.prev_pos_for_vel = self.last_orientation
            self.prev_pos_for_vel[:3, 3] = self.last_position

    def gather_data(self):
        with self.lock:
            self.pinned_pose = self.last_orientation
            self.pinned_pose[:3, 3] = self.last_position

    def step(self):
        if self.last_position is not None and self.last_orientation is not None:
            if (
                self.last_save_t is None
                or rospy.Time.now() - self.last_save_t > self.time_resolution
            ):
                self.last_save_t = rospy.Time.now()
                self.gather_data()


class PointAndShootServer:
    def __init__(self, accumulator: CameraAccumulator) -> None:
        self.server = actionlib.SimpleActionServer(
            "bc", GoToGoalBCAction, self.execute, False
        )
        self.accumulator = accumulator
        self.pub = rospy.Publisher(
            "/husky_velocity_controller/cmd_vel",
            Twist,
            queue_size=1,
        )
        self.timeout = rospy.Duration(10)
        self.goal_tolerance = 0.5
        self.goal_max_distance = 5.0  # Just to avoid going too far
        self.server.start()

    def execute(self, goal: GoToGoalBCActionGoal):
        # rospy.logwarn(f"Received goal {goal}")
        goal: Point = goal.goal
        goal = np.array([goal.x, goal.y, goal.z, 1.0])
        begin = rospy.Time.now()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            pose = self.accumulator.pinned_pose
            goal_local_frame = np.linalg.inv(pose) @ goal

            # Checking termination conditions
            if np.linalg.norm(goal_local_frame[:2]) < self.goal_tolerance:
                rospy.logwarn("SUCCESS")
                result = GoToGoalBCResult()
                result.success_code.data = 0
                self.server.set_succeeded(result)
                return result
            elif np.linalg.norm(goal_local_frame[:2]) > self.goal_max_distance:
                rospy.logwarn("TOO FAR")
                result = GoToGoalBCResult()
                result.success_code.data = 1
                self.server.set_aborted(result)
                return result
            elif rospy.Time.now() - begin > self.timeout:
                rospy.logwarn("TOO LONG")
                result = GoToGoalBCResult()
                result.success_code.data = 1
                self.server.set_aborted(result)
                return result
            goal_local_frame = goal_local_frame[:3]
            angle = np.arctan2(goal_local_frame[1], goal_local_frame[0])
            if abs(angle) > 0.4:
                vx = 0
                vyaw = 0.4 * np.sign(angle)
            else:
                vx = 0.5
                vyaw = 0
            msg = Twist()
            msg.linear.x = vx
            msg.linear.y = 0
            msg.linear.z = 0
            msg.angular.x = 0
            msg.angular.y = 0
            msg.angular.z = vyaw
            self.pub.publish(msg)
            rate.sleep()


def spin_acc(acc):
    rate = rospy.Rate(15)
    while not rospy.is_shutdown():
        acc.step()
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node("bc_node")
    acc = CameraAccumulator(5, False)
    spin_acc_closure = lambda: spin_acc(acc)
    spin_acc_thread = Thread(target=spin_acc_closure)
    spin_acc_thread.start()
    server = PointAndShootServer(acc)
    rospy.spin()
