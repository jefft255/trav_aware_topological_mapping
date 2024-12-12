import numpy as np
from topological_mapping.topological_map import (
    TopologicalMap,
    MapEdge,
    MapNode,
    RealMapNode,
)
from copy import deepcopy


class TopologicalSimulator:
    def __init__(self, full_map: TopologicalMap, edge_callback):
        self.full_map = full_map
        self.edge_callback = edge_callback

    def init(self):
        self.simulated_map = TopologicalMap(
            self.full_map.resolution,
            self.full_map.grid_resolution,
            self.full_map.nodes[0],
            self.edge_callback,
        )
        return self.simulated_map.current_node

    def find_actually_reached_node(self, virtual_node: MapNode) -> RealMapNode:
        grid_index_virtual = self.simulated_map.occupancy_map.get_buffer_coordinates(
            virtual_node.translation[:2]
        )
        print(f"{grid_index_virtual=}")
        for n in self.full_map.nodes:
            if not isinstance(n, RealMapNode):
                continue
            grid_index_real = self.simulated_map.occupancy_map.get_buffer_coordinates(
                n.translation[:2]
            )
            if grid_index_real == grid_index_virtual:
                print(f"{grid_index_real=}")
                return n

        raise ValueError("Can't find a real node corresponding to the frontier")

    def step(self, n: MapNode):
        print(f"{self.simulated_map.current_node.translation=}")
        assert n in self.simulated_map.current_node.reachable_neighbors
        actual_reached_node = self.find_actually_reached_node(n)

        print(f"{actual_reached_node.translation=}")
        # Don't use deepcopy because we're throwing away (un)reachle
        # neighbors pointers
        closest_node_dist, closest_node_id = self.simulated_map.twod_kdtree.query(
            actual_reached_node.translation[:2],
            k=1,
        )
        print(f"{closest_node_dist=}")
        print(f"{closest_node_id=}")

        if closest_node_dist < 0.01:
            print("Node already in map")
            self.simulated_map.current_node = self.simulated_map.nodes[closest_node_id]
        else:
            print("Node not in map, adding")
            actual_reached_node = RealMapNode(
                actual_reached_node.image_l,
                actual_reached_node.image_f,
                actual_reached_node.image_r,
                actual_reached_node.translation,
                actual_reached_node.rotation,
                actual_reached_node.t,
            )
            self.simulated_map.add_node(actual_reached_node)
        return self.simulated_map.current_node
