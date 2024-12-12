#!/usr/bin/env python3
import tkinter as tk
from tkinter import Misc, ttk, Canvas
from typing import Any
from typing_extensions import Literal
import numpy as np
from pathlib import Path
import rospy
from threading import Thread, Lock
import signal
from datetime import datetime
import os

from topological_mapping.msg import GoToGoalBCAction, GoToGoalBCGoal
from actionlib import SimpleActionClient
from actionlib import SimpleGoalState
from tf.transformations import quaternion_from_matrix
from visualization_msgs.msg import Marker


from topological_mapping.map_builder import MapBuilder
from topological_mapping.topological_map import (
    MapNode,
    TopologicalMap,
    Traversability,
    MapEdge,
)
from topological_mapping.planner import AStarPlanner
from topological_mapping.viz import TopologicalMapRViz, create_bc_marker_from_node
from topological_mapping.exploration_planner import (
    ClosestFrontierSelector,
    KinematicallyClosestFrontierSelector,
)
from topological_mapping.srv import MapSaver, MapSaverRequest, MapSaverResponse


def save_handler(map: TopologicalMap, req: MapSaverRequest):
    rospy.logwarn("Saving map...")
    path = Path(req.path.data)
    map.save(path)
    rospy.logwarn("Saving map done!")
    response = MapSaverResponse(0)
    return response


def create_goal_from_node(node: MapNode) -> GoToGoalBCGoal:
    # Goal is assumed to be in odom frame
    # Or actually in whatever frame /odometry/filtered is
    goal = GoToGoalBCGoal()
    goal.goal.x = node.translation[0]
    goal.goal.y = node.translation[1]
    goal.goal.z = node.translation[2]
    return goal


def save_traversability_result(nodeA: MapNode, nodeB: MapNode, result, run_path: Path):
    save_folder = run_path / "traversability_dataset"
    save_folder.mkdir(exist_ok=True)

    edges_folders = list(save_folder.glob("*"))
    if len(edges_folders) == 0:
        id = 0
    else:
        id = max([int(e.stem) for e in edges_folders]) + 1
    save_folder = save_folder / str(id)
    save_folder.mkdir()

    if result == 3:
        trav = Traversability.TRAVERSABLE
    else:
        trav = Traversability.UNTRAVERSABLE
    edge = MapEdge(nodeA, nodeB, trav)
    nodeA.save(None, save_folder, 0)
    nodeB.save(None, save_folder, 1)
    edge.save(None, save_folder, 0)


def exploration_thread_f(map_builder: MapBuilder, run_path: Path, map_lock: Lock):
    map = map_builder.map
    frontier_selector = KinematicallyClosestFrontierSelector(map)
    action_client = SimpleActionClient("bc", GoToGoalBCAction)
    marker_pub = rospy.Publisher("/bc/viz_marker", Marker, queue_size=1)

    while not rospy.is_shutdown():
        with map_lock:
            map.grow_map()
        frontier_goal = frontier_selector.get_frontier()
        if frontier_goal is None:
            rospy.logwarn("No more frontiers to explore!")
            map_save_path = run_path / "map"
            map_save_path.mkdir(exist_ok=True)
            map.save(map_save_path)
            rospy.sleep(1.0)
            continue

        # if False:
        rospy.logwarn("Planning...")
        planner = AStarPlanner(map, frontier_goal)
        node_at_start = map.current_node
        plan = planner.plan(node_at_start)

        if plan is None:
            rospy.logerr(
                "No plan found! Something is broken because the node is reachable."
            )
            rospy.sleep(1.0)
            continue

        action_client.wait_for_server()

        node = plan[0]

        # for node in plan:
        rospy.logwarn(f"!!!!!!!!!!!!!!!!! Going to {node}... !!!!!!!!!!!!!!")
        goal = create_goal_from_node(node)
        marker = create_bc_marker_from_node(node)
        marker_pub.publish(marker)
        action_client.send_goal_and_wait(
            goal, rospy.Duration(10000), rospy.Duration(10000)
        )
        # while action_client.simple_state is not SimpleGoalState.DONE:
        #     rospy.sleep(0.3)
        status = action_client.gh.get_goal_status()
        rospy.logwarn(f"{action_client.gh.get_goal_status()}")
        if status == 3:
            rospy.logwarn("Goal reached successfully")
        elif status == 4:
            pass
            # rospy.logwarn("Goal not reached! Saving...")
            # try:
            #     map.relabel_edge(
            #         None, node_at_start, node, Traversability.UNTRAVERSABLE
            #     )
            # except ValueError:
            #     rospy.logwarn("Frontier has disappeared; still saving result")
        else:
            raise ValueError("Goal status is not SUCCEEDED or ABORTED")
        save_traversability_result(node_at_start, node, status, run_path)
        with map_lock:
            map.add_node(map_builder.gather_node_data())
        rospy.logwarn(f"----------- Done going to {node} -----------")
        # elif True:
        #     rospy.sleep(1.0)


if __name__ == "__main__":
    rospy.init_node("map_builder")
    rate = rospy.Rate(10)
    viz_time = rospy.Duration(2.0)
    last_viz_time = rospy.Time.now()
    map_builder = MapBuilder(1.0, 2.0, 3.0)

    now = datetime.now()
    run_path = Path(os.getenv("HOME")) / Path(f"{now.strftime('%Y-%m-%d-%H-%M-%S')}")
    run_path.mkdir()

    plan = None
    map_viz = None
    exploration_thread = None
    save_handler_closure = None
    saver_service = None
    map_lock = Lock()

    rospy.logwarn("Drive around for 20s to get heading to converge.")
    rospy.sleep(20)

    while not rospy.is_shutdown():
        with map_lock:
            map_builder.step()
        if map_builder.map is not None:
            if save_handler_closure is None:

                def save_handler_closure(x):
                    return save_handler(map_builder.map, x)

                saver_service = rospy.Service(
                    "map_saver", MapSaver, handler=save_handler_closure
                )

            if exploration_thread is None:
                exploration_thread = Thread(
                    target=exploration_thread_f, args=(map_builder, run_path, map_lock)
                )
                exploration_thread.start()

            if map_viz is None:
                map_viz = TopologicalMapRViz(map_builder.map)
            elif rospy.Time.now() - last_viz_time > viz_time:
                map_viz.refresh()
                last_viz_time = rospy.Time.now()
