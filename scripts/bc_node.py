#!/usr/bin/env python
import rospy
import actionlib
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
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

from topological_mapping.learning_bc.models import DINOv2BCNet, ResNetBCNet

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
    def __init__(self, hist_size, only_front, device) -> None:
        self.bridge = CvBridge()
        self.hist_size = hist_size
        self.only_front = only_front
        self.last_image_f: Image = None
        self.last_image_l: Image = None
        self.last_image_r: Image = None
        self.last_cmd: Twist = Twist()
        self.last_command: Optional[npt.NDArray[np.float32]] = None
        self.last_position: Optional[npt.NDArray[np.float64]] = None
        self.last_position_t = Optional[rospy.Time]
        self.first_position: Optional[npt.NDArray[np.float64]] = None
        self.last_orientation: Optional[npt.NDArray[np.float64]] = None
        self.prev_pos_for_vel: Optional[npt.NDArray[np.float64]] = None
        self.image_subs = [
            rospy.Subscriber(f"/oak_{ori}/color/image_raw", Image, callback)
            for ori, callback in zip(
                ["front", "left", "right"],
                [self.image_callback_f, self.image_callback_l, self.image_callback_r],
            )
        ]
        self.odom_subscriber = rospy.Subscriber(
            "/odometry/filtered", Odometry, self.odom_callback
        )
        self.cmd_subscriber = rospy.Subscriber(
            "/husky_velocity_controller/cmd_vel", Twist, self.cmd_callback
        )
        self.cmd_list = self.hist_size * [torch.zeros((2,), device=device)]
        self.image_l_list = []
        self.image_f_list = []
        self.image_r_list = []
        self.vel_list = []
        self.pinned_pose = None
        self.time_resolution = rospy.Duration(0.2)
        self.last_save_t = None
        self.transform = ToTensor()
        self.lock = Lock()
        self.device = device

    def cmd_callback(self, cmd_msg: Twist):
        self.last_cmd = cmd_msg

    def image_callback_f(self, img_msg: Image):
        self.last_image_f = img_msg
        if self.only_front:
            # Just to skip the existence check later on
            self.last_image_l = img_msg
            self.last_image_r = img_msg

    def image_callback_l(self, img_msg: Image):
        self.last_image_l = img_msg

    def image_callback_r(self, img_msg: Image):
        self.last_image_r = img_msg

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

    def process_image(self, img: Image):
        img = cv2.resize(
            self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8"), (224, 224)
        )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img).to(device=self.device)
        return img

    def gather_data(self):
        with self.lock:
            self.image_f_list.append(self.process_image(self.last_image_f))
            self.image_l_list.append(self.process_image(self.last_image_l))
            self.image_r_list.append(self.process_image(self.last_image_r))
            if len(self.image_f_list) > self.hist_size:
                self.image_f_list = self.image_f_list[-self.hist_size :]
                self.image_l_list = self.image_l_list[-self.hist_size :]
                self.image_r_list = self.image_r_list[-self.hist_size :]
            self.cmd_list.append(
                torch.tensor(
                    [self.last_cmd.linear.x, self.last_cmd.angular.z],
                    device=self.device,
                )
            )
            self.cmd_list = self.cmd_list[-self.hist_size :]
            self.pinned_pose = self.last_orientation
            self.pinned_pose[:3, 3] = self.last_position
            self.vel_list.append(
                torch.tensor(
                    (np.linalg.inv(self.prev_pos_for_vel) @ self.pinned_pose)[:3],
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            if len(self.vel_list) > self.hist_size:
                self.vel_list = self.vel_list[-self.hist_size :]
            self.prev_pos_for_vel = self.pinned_pose

    def step(self):
        if (
            self.last_position is not None
            and self.last_image_f is not None
            and self.last_image_l is not None
            and self.last_image_r is not None
            and self.last_orientation is not None
        ):
            if (
                self.last_save_t is None
                or rospy.Time.now() - self.last_save_t > self.time_resolution
            ):
                self.last_save_t = rospy.Time.now()
                self.gather_data()


class BCServer:
    def __init__(self, model, accumulator: CameraAccumulator, device) -> None:
        self.server = actionlib.SimpleActionServer(
            "bc", GoToGoalBCAction, self.execute, False
        )
        self.accumulator = accumulator
        self.pub = rospy.Publisher(
            "/husky_velocity_controller/cmd_vel",
            # "/joy_teleop/cmd_vel",
            Twist,
            queue_size=1,
        )
        self.model = model
        self.timeout = rospy.Duration(10)
        self.goal_tolerance = 0.9
        self.goal_max_distance = 5.0  # Just to avoid going too far
        self.device = device
        self.server.start()

    def execute(self, goal: GoToGoalBCActionGoal):
        # rospy.logwarn(f"Received goal {goal}")
        goal: Point = goal.goal
        goal = np.array([goal.x, goal.y, goal.z, 1.0])
        begin = rospy.Time.now()
        with torch.no_grad():
            rate = rospy.Rate(10)
            while not rospy.is_shutdown():
                begin_inf = rospy.Time.now()
                with self.accumulator.lock:
                    img_l = self.accumulator.image_l_list
                    img_f = self.accumulator.image_f_list
                    img_r = self.accumulator.image_r_list
                    past_cmds = self.accumulator.cmd_list
                    vels = self.accumulator.vel_list
                    if len(img_l) < self.accumulator.hist_size:
                        continue
                    # Stack over time and add batch dim
                    img_l = torch.stack(img_l).unsqueeze(dim=0)
                    img_f = torch.stack(img_f).unsqueeze(dim=0)
                    img_r = torch.stack(img_r).unsqueeze(dim=0)
                    vels = torch.stack(vels).unsqueeze(dim=0)
                    past_cmds = torch.stack(past_cmds).unsqueeze(dim=0)
                    past_cmds = (
                        past_cmds - self.model.mu.to(self.device)
                    ) / self.model.std.to(self.device)
                pose = self.accumulator.pinned_pose
                goal_local_frame = np.linalg.inv(pose) @ goal
                # rospy.logwarn(f"Goal in local frame is {goal_local_frame}")
                # rospy.logwarn(f"Goal in global frame is {goal}")
                # rospy.logwarn(f"Pose is {pose}")
                # rospy.logwarn(
                #     f"Distance to goal is {np.linalg.norm(goal_local_frame[:3])}"
                # )

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
                goal_local_frame = torch.tensor(
                    goal_local_frame[:3].astype(np.float32), device=self.device
                ).unsqueeze(dim=0)
                model_out = self.model(
                    img_l, img_f, img_r, vels, past_cmds, goal_local_frame
                )
                model_out = self.model.std.to(
                    self.device
                ) * model_out + self.model.mu.to(self.device)
                # rospy.logwarn(f"Output is {model_out}")
                model_out = model_out[
                    :, 0
                ]  # Only execute first action (MPC style but for BC)
                end_inf = rospy.Time.now()
                # rospy.logwarn(f"Took {(end_inf - begin_inf).to_sec()}")
                assert model_out.shape == torch.Size((1, 2))
                vx = model_out[0, 0].item()
                vyaw = model_out[0, 1].item()
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

    if torch.cuda.device_count() > 1:
        print("Multiple GPUs, running BC on second one")
        device = torch.device("cuda:1")
    else:
        device = torch.device("cuda")
    torch.cuda.init()
    myargv = rospy.myargv(sys.argv)

    if (Path(myargv[1]) / "model_bc.pth").exists():
        model_path = Path(myargv[1]) / "model_bc.pth"
    elif (Path(myargv[1]) / "model_bc_dino.pth").exists():
        model_path = Path(myargv[1]) / "model_bc_dino.pth"
    elif (Path(myargv[1]) / "model_bc_resnet.pth").exists():
        model_path = Path(myargv[1]) / "model_bc_resnet.pth"
    else:
        raise FileNotFoundError("No model found")
    print(model_path)
    model = torch.load(open(model_path, "rb"), map_location=device)
    model.to(device)
    model.eval()
    acc = CameraAccumulator(model.hist_size, model.only_front, device)
    spin_acc_closure = lambda: spin_acc(acc)
    spin_acc_thread = Thread(target=spin_acc_closure)
    spin_acc_thread.start()
    server = BCServer(model, acc, device)
    rospy.spin()
