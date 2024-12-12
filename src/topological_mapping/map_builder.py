#!/home/jft/catkin_ws/src/topological_mapping_venv/bin/python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from tf.transformations import quaternion_matrix

from cv_bridge import CvBridge

import numpy as np
import numpy.typing as npt
from typing import Optional, List

from topological_mapping.topological_map import (
    TopologicalMap,
    RealMapNode,
    Traversability,
)
from topological_mapping.utils import project_node
from topological_mapping.srv import TraversabilityAnalyzer


class MapBuilder:
    def __init__(
        self, dist_to_new_node: float, grid_resolution: float, neighbourhood_size: float
    ) -> None:
        self.bridge = CvBridge()
        self.dist_to_new_node = dist_to_new_node
        self.grid_resolution = grid_resolution
        self.neighbourhood_size = neighbourhood_size
        self.last_image_f: Image = None
        self.last_image_l: Image = None
        self.last_image_r: Image = None
        self.last_position: Optional[npt.NDArray[np.float64]] = None
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
        self.map = None
        self.traversability_service = rospy.ServiceProxy(
            "traversability_analyser", TraversabilityAnalyzer
        )
        rospy.wait_for_service("traversability_analyser")

    def traversability_callback(self, nodeA, nodeB):
        # return Traversability.TRAVERSABLE
        imgA, _, Pa_3d, pixel = project_node(nodeA, nodeB)
        if imgA is None and isinstance(nodeB, RealMapNode):
            imgA, _, Pa_3d, pixel = project_node(nodeB, nodeA)
        if imgA is None:
            # No co-visiblity
            return Traversability.UNKNOWN  # No co-visibility
        imgA = self.bridge.cv2_to_imgmsg(imgA)
        position = Point()
        position.x = Pa_3d[0]
        position.y = Pa_3d[1]
        position.z = Pa_3d[2]
        pixel_x = Int32()
        pixel_x.data = int(pixel[0])
        pixel_y = Int32()
        pixel_y.data = int(pixel[1])
        response = self.traversability_service(imgA, position, pixel_x, pixel_y)
        if response.response.data == 0:
            return Traversability.TRAVERSABLE
        elif response.response.data == 1:
            return Traversability.UNTRAVERSABLE
        else:
            raise ValueError("")

    def image_callback_f(self, img_msg: Image):
        self.last_image_f = img_msg

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

    def step(self):
        if (
            self.last_position is not None
            and self.last_image_f is not None
            and self.last_image_l is not None
            and self.last_image_r is not None
            and self.last_orientation is not None
        ):
            if self.map is None:
                rospy.logwarn("Creating map")
                self.map = TopologicalMap(
                    self.neighbourhood_size,
                    self.grid_resolution,
                    self.gather_node_data(),
                    self.traversability_callback,
                )
                self.last_node_position = self.last_position
            if (
                np.linalg.norm(self.last_position - self.last_node_position)
                >= self.dist_to_new_node
            ):
                rospy.logwarn(
                    f"Adding node at {self.last_position - self.first_position}"
                )
                rospy.logwarn(
                    f"Distance is {np.linalg.norm(self.last_position - self.last_node_position)}"
                )
                self.map.add_node(self.gather_node_data())
                self.last_node_position = self.last_position
