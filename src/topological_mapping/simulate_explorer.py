from topological_mapping.topological_map import (
    TopologicalMap,
    MapEdge,
    MapNode,
    RealMapNode,
    Traversability,
)
from topological_mapping.exploration_planner import ClosestFrontierSelector
from topological_mapping.topological_simulator import TopologicalSimulator
from topological_mapping.planner import AStarPlanner
from topological_mapping.utils import project_node
from topological_mapping.srv import TraversabilityAnalyzer


from cv_bridge import CvBridge
from geometry_msgs.msg import Point
import sys
from pathlib import Path
import rospy


if __name__ == "__main__":
    traversability_service = rospy.ServiceProxy(
        "traversability_analyser", TraversabilityAnalyzer
    )
    rospy.wait_for_service("traversability_analyser")
    bridge = CvBridge()

    def traversability_callback(nodeA, nodeB):
        return Traversability.TRAVERSABLE
        imgA, _, Pa_3d, _ = project_node(nodeA, nodeB)
        if imgA is None and isinstance(nodeB, RealMapNode):
            imgA, _, Pa_3d, _ = project_node(nodeB, nodeA)
        if imgA is None:
            return Traversability.UNKNOWN  # No co-visibility
        imgA = bridge.cv2_to_imgmsg(imgA)
        position = Point()
        position.x = Pa_3d[0]
        position.y = Pa_3d[1]
        position.z = Pa_3d[2]
        response = traversability_service(imgA, position)
        if response.response.data == 0:
            return Traversability.TRAVERSABLE
        elif response.response.data == 1:
            return Traversability.UNTRAVERSABLE
        else:
            raise ValueError("")

    full_map = TopologicalMap.load(Path(sys.argv[1]))
    print(f"{full_map.nodes[0].translation=}")
    sim = TopologicalSimulator(full_map, traversability_callback)
    current_node = sim.init()
    selector = ClosestFrontierSelector(sim.simulated_map)

    while True:
        sim.simulated_map.grow_map()
        target_node = selector.get_frontier()
        print(f"goal is {target_node.translation}")
        print(f"Current node is {sim.simulated_map.current_node.translation}")
        planner = AStarPlanner(sim.simulated_map, target_node)
        plan = planner.plan(current_node)
        if plan is None:
            print("No plan found! Something is broken because the node is reachable.")
        for node in plan:
            print(f"Going to {node}...")
            current_node = sim.step(node)
