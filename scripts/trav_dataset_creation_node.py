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
        self.last_position: Optional[npt.NDArray[np.float64]] = None
        self.last_node_position: npt.NDArray[np.float64] = np.array(
            [-np.inf, -np.inf, -np.inf]
        )
        self.last_position_t = Optional[rospy.Time]
        self.first_position: Optional[npt.NDArray[np.float64]] = None
        self.traj: List[npt.NDArray[np.float64]] = []
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
        self.node_list = []
        self.last_image_t_nonros = None
        self.dist_to_new_node = 1.0

    def image_callback_f(self, img_msg: Image):
        self.last_image_f = img_msg
        self.last_image_t_nonros = time()

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
        if self.first_position is None:
            self.first_position = np.array(
                [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ]
            )
        self.traj.append(self.last_position)
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
        return self.bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")

    def gather_node_data(self):
        node = RealMapNode(
            self.process_image(self.last_image_l),
            self.process_image(self.last_image_f),
            self.process_image(self.last_image_r),
            self.last_position,
            self.last_orientation,
            self.last_position_t.to_sec(),
        )
        return node

    def save(self):
        for i, n in enumerate(self.node_list):
            n.save(None, self.save_path, i)

    def step(self):
        if self.last_image_t_nonros is not None:
            if time() - self.last_image_t_nonros > 20.0:
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
        ):
            if self.last_node_position is None or (
                np.linalg.norm(self.last_position - self.last_node_position)
                >= self.dist_to_new_node
            ):
                rospy.logwarn(
                    f"Adding node at {self.last_position - self.first_position}"
                )
                rospy.logwarn(
                    f"Distance is {np.linalg.norm(self.last_position - self.last_node_position)}"
                )
                self.last_node_position = self.last_position
                self.node_list.append(self.gather_node_data())


if __name__ == "__main__":
    rospy.init_node("dataset_builder")
    save_path = Path(sys.argv[1])
    rate = rospy.Rate(100)
    map_builder = DatasetBuilder(save_path)

    while not rospy.is_shutdown():
        map_builder.step()
        # rate.sleep()
