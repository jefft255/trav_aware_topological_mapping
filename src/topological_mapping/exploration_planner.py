# Goals:
# Create a class that receives as input the current node
# and outputs the closest unvisited node

from typing import List
import numpy as np
from topological_mapping.topological_map import (
    TopologicalMap,
    MapNode,
)
import rospy


class ClosestFrontierSelector:
    def __init__(self, topological_map: TopologicalMap):
        # First step here is to get all the frontier nodes that are not visited
        self.t_map = topological_map

    def get_frontier(self):
        # Second step is to calculate the norm between the current node and the frontier
        # iterate through all of them, while having a variable named min_norm set to infinity
        frontiers: List[MapNode] = self.t_map.get_reachable_frontiers()
        min_norm = np.inf
        closest_frontier = None

        for frontier in frontiers:
            current_norm = np.linalg.norm(
                self.t_map.current_node.translation - frontier.translation
            )
            if current_norm < min_norm:
                min_norm = current_norm
                closest_frontier = frontier

        # Return the closest frontier
        rospy.logwarn(f"Closest frontier: {min_norm}")
        return closest_frontier


class KinematicallyClosestFrontierSelector:
    def __init__(self, topological_map: TopologicalMap):
        self.t_map = topological_map
        self.max_rot_speed_rads = 0.3
        self.max_lin_speed_ms = 1.0

    def get_frontier(self):
        frontiers: List[MapNode] = self.t_map.get_reachable_frontiers()
        min_norm = np.inf
        closest_frontier = None

        for frontier in frontiers:
            lin_dist = np.linalg.norm(
                self.t_map.current_node.translation - frontier.translation
            )
            lin_dist_s = lin_dist / self.max_lin_speed_ms

            robot_x_vector = self.t_map.current_node.rotation[:3, 0]
            frontier_vector = frontier.translation - self.t_map.current_node.translation
            frontier_vector /= lin_dist
            frontier_angle = np.arccos(np.dot(robot_x_vector, frontier_vector))

            ang_dist_s = frontier_angle / self.max_rot_speed_rads

            current_norm = lin_dist_s + ang_dist_s

            if current_norm < min_norm:
                min_norm = current_norm
                closest_frontier = frontier

        # Return the closest frontier
        rospy.logwarn(f"Closest frontier: {min_norm} seconds")
        return closest_frontier
