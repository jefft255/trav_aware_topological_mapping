from tqdm import tqdm
import rospy
import numpy as np
from itertools import product
from geometry_msgs.msg import PoseStamped
from topological_mapping.topological_map import (
    TopologicalMap,
    RealMapNode,
    MapEdge,
    MapNode,
)
import heapq


class MoveBaseController:
    def __init__(self):
        self.pub = rospy.Publisher("/move_base_simple/goal", PoseStamped)


class PlanningNode:
    def __init__(self, cost: float, node: MapNode, parent):
        self.cost = cost
        self.node = node
        self.parent = parent

    def __lt__(self, other):
        return self.cost < other.cost


class DijkstraPlanner:
    def __init__(self, map, goal):
        self.map = map
        self.goal = goal

    def get_neighbors(self, node: PlanningNode):
        # Return a list of PlanningNodes for each neighbor
        return [
            PlanningNode(
                node.cost + 1.0, neighbor, node
            )  # Assuming a uniform cost for simplicity
            for neighbor in node.node.reachable_neighbors
        ]

    def plan(self, start: MapNode):
        open_list = []
        closed_list = set()

        # Initialize the goal node with cost 0 and add to the open list
        goal_node = PlanningNode(0, self.goal, None)
        heapq.heappush(open_list, (0, goal_node))
        node_costs = {self.goal: 0}  # Dictionary to track the cost for each node

        while open_list:
            current_cost, current_node = heapq.heappop(open_list)

            if current_node.node in closed_list:
                continue

            # Mark the node as visited
            closed_list.add(current_node.node)

            # For each neighbor, calculate the new cost and update if lower
            for neighbor in self.get_neighbors(current_node):
                if neighbor.node not in closed_list:
                    new_cost = (
                        current_cost + 1.0
                    )  # Uniform cost (can be modified for more complex cost models)
                    if (
                        neighbor.node not in node_costs
                        or new_cost < node_costs[neighbor.node]
                    ):
                        node_costs[neighbor.node] = new_cost
                        heapq.heappush(open_list, (new_cost, neighbor))

        # Now `node_costs` contains the minimum cost-to-goal for each node
        return node_costs


class AStarPlanner:
    def __init__(self, map, goal):
        self.map = map
        self.goal = goal
        self.cost = 1.0  # Constant cost for now; would be interesting to modify

    def heuristic(self, node: PlanningNode):
        # Probably admissible heuristic for distance cost;
        return np.linalg.norm(node.node.translation - self.goal.translation)
        # When doing other costs, this silly heuristic should be used but it's super slow.
        # return -np.inf

    def get_neighbors(self, node: PlanningNode):
        return [
            PlanningNode(node.cost + self.cost, neighbor, node)
            for neighbor in node.node.reachable_neighbors
        ]

    def plan(self, start: MapNode):
        open_list = []
        closed_list = set()

        start = PlanningNode(0, start, None)
        heapq.heappush(open_list, (0, start))

        while open_list:
            current_cost, current_node = heapq.heappop(open_list)

            if current_node.node is self.goal:
                # Goal reached, construct and return the path
                path = []
                while current_node:
                    path.append(current_node.node)
                    current_node = current_node.parent
                path = path[::-1]
                path = path[1:]  # Remove the start node
                return path

            closed_list.add(current_node)

            for neighbor in self.get_neighbors(current_node):
                if neighbor in closed_list:
                    continue

                new_cost = current_node.cost + self.cost
                neighbor_is_in_open_list = False
                neighbor_id_in_open_list = -1
                for i, (_, n) in enumerate(open_list):
                    if n.node is neighbor.node:
                        neighbor_is_in_open_list = True
                        neighbor_id_in_open_list = i
                if not neighbor_is_in_open_list:
                    heapq.heappush(
                        open_list,
                        (
                            new_cost + self.heuristic(neighbor),
                            neighbor,
                        ),
                    )
                elif new_cost < neighbor.cost:
                    open_list[neighbor_id_in_open_list][1].cost = new_cost
                    open_list[neighbor_id_in_open_list][1].parent = current_node


if __name__ == "__main__":
    print("Testing graph algorithm")
    print("Creating map...")
    map = TopologicalMap(
        1.0,
        RealMapNode(
            None,
            None,
            None,
            None,
            np.array(
                [
                    -1,
                    0,
                    0,
                ]
            ),
            np.eye(3),
            0,
        ),
    )
    n = 102
    xs, ys = np.meshgrid(
        np.linspace(0, n, n + 2),
        np.linspace(0, n, n + 2),
    )
    for i in range(xs.shape[0]):
        js = range(xs.shape[1]) if i % 2 == 0 else range(xs.shape[1] - 1, -1, -1)
        for j in js:
            map.add_node(
                RealMapNode(
                    None,
                    None,
                    None,
                    None,
                    np.array(
                        [
                            xs[i, j],
                            ys[i, j],
                            0,
                        ]
                    ),
                    np.eye(3),
                    0,
                ),
            )

    print(f"{len(map.edges)} edges")
    print(f"{len(map.nodes)} nodes")
    print("Planning...")
    astar = AStarPlanner(map, map.nodes[-1])
    path = astar.plan(map.nodes[0])
    for n in path:
        print(n.translation)
    import pickle
    import sys

    sys.setrecursionlimit(10000000)

    pickle.dump(map, open("/home/adaptation/jft/map.pickle", "wb"))
