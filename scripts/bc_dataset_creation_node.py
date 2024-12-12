#!/home/jft/catkin_ws/src/topological_mapping_venv/bin/python
from typing import Any, Optional, List
from typing_extensions import Literal
import pickle
from time import time
from pathlib import Path
import sys

from topological_mapping.topological_map import RealMapNode

import rospy
from sensor_msgs.msg import NavSatFix, Image, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import quaternion_matrix
from cv_bridge import CvBridge
import cv2

import numpy as np
import numpy.typing as npt


class DatasetBuilder:
    def __init__(self, save_path: Path) -> None:
        self.save_path = save_path
        self.bridge = CvBridge()
        self.last_image_f: Image = None
        self.last_image_l: Image = None
        self.last_image_r: Image = None
        self.last_command: Optional[npt.NDArray[np.float32]] = None
        self.last_position: Optional[npt.NDArray[np.float64]] = None
        self.last_position_t = Optional[rospy.Time]
        self.first_position: Optional[npt.NDArray[np.float64]] = None
        self.last_orientation: Optional[npt.NDArray[np.float64]] = None  # TODO
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
        self.control_subscriber = rospy.Subscriber(
            "/husky_velocity_controller/cmd_vel", Twist, self.control_callback
        )
        self.image_l_list = []
        self.image_f_list = []
        self.image_r_list = []
        self.command_list = []
        self.position_list = []
        self.orientation_list = []
        self.time_resolution = rospy.Duration(0.2)
        self.last_save_t = None
        self.last_save_t_nonros = None
        self.last_save_t_nonros_image = None

    def image_callback_f(self, img_msg: Image):
        self.last_image_f = img_msg
        self.last_save_t_nonros_image = time()

    def image_callback_l(self, img_msg: Image):
        self.last_image_l = img_msg

    def image_callback_r(self, img_msg: Image):
        self.last_image_r = img_msg

    def control_callback(self, ctrl_msg: Twist):
        self.last_command = np.array(
            [ctrl_msg.linear.x, ctrl_msg.angular.z], dtype=np.float32
        )

    def odom_callback(self, msg: Odometry):
        self.last_position = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
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
        self.last_position_t = msg.header.stamp

        self.last_orientation = quaternion_matrix(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ]
        )

    def process_image(self, img: Image):
        return cv2.resize(
            self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8"), (224, 224)
        )

    def gather_data(self):
        self.image_f_list.append(self.process_image(self.last_image_f))
        self.image_l_list.append(self.process_image(self.last_image_l))
        self.image_r_list.append(self.process_image(self.last_image_r))
        self.position_list.append(self.last_position)
        self.orientation_list.append(self.last_orientation)
        self.command_list.append(self.last_command)

    def save(self):
        pickle.dump(
            {
                "images_f": self.image_f_list,
                "images_l": self.image_l_list,
                "images_r": self.image_r_list,
                "positions": self.position_list,
                "orientations": self.orientation_list,
                "commands": self.command_list,
            },
            open(self.save_path, "wb"),
        )

    def step(self):
        if self.last_save_t_nonros_image is not None:
            if time() - self.last_save_t_nonros_image > 10.0:
                print("No new data, saving...")
                self.save()
                print("Done!")
                exit(0)
        if (
            self.last_position is not None
            and self.last_image_f is not None
            and self.last_image_l is not None
            and self.last_image_r is not None
            and self.last_orientation is not None
            and self.last_command is not None
        ):
            if (
                self.last_save_t is None
                or rospy.Time.now() - self.last_save_t > self.time_resolution
            ):
                self.last_save_t = rospy.Time.now()
                self.last_save_t_nonros = time()
                print(len(self.image_f_list))
                self.gather_data()


if __name__ == "__main__":
    rospy.init_node("dataset_builder")
    save_path = Path(sys.argv[1])
    rate = rospy.Rate(100)
    map_builder = DatasetBuilder(save_path)

    while not rospy.is_shutdown():
        map_builder.step()
        # rate.sleep()
