from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Optional, List, Callable
from scipy.spatial import KDTree
from enum import Enum, auto
import json
from pathlib import Path
import cv2
from tqdm import tqdm
from topological_mapping.occupancy_map import OccupancyMap
import rospy


ImageT = Optional[npt.NDArray[np.uint8]]
Translation = npt.NDArray[np.float64]  # Need double precision when working with GPS
Rotation = npt.NDArray[np.float64]  # Need double precision when working with GPS
# (in secs) could be ROS time but want to avoid any existence of ROS here!
TimeT = float


def MapNodeFactory(path: Path) -> MapNode:
    image_path = path / "front.png"
    if image_path.exists():
        data = json.load(open(path / "node.json", "r"))
        if "rotation" in data.keys():
            return RealMapNode.load(path)
        else:
            print(f"Warning: images found but no rotation data for node at {path}")
            return MapNode.load(path)
    else:
        return MapNode.load(path)


class MapNode:
    def __init__(self, translation: Translation) -> None:
        self.translation = translation
        self.reachable_neighbors: List[MapNode] = []
        self.unreachable_neighbors: List[MapNode] = []

    @property
    def virtual(self):
        return True

    def __str__(self):
        return f"Node at {self.translation}"

    def serialize(self, path: Path, id: int):
        return {
            "type": self.__class__.__name__,
            "translation": self.translation.tolist(),
        }

    def save(
        self, tmap: Optional[TopologicalMap], path: Path, id: Optional[int] = None
    ):
        # assert (tmap is not None) != (id is not None)
        if tmap is not None:
            id = tmap.get_node_id(self)
        out = json.dumps(self.serialize(path, id))
        output_path = path / "nodes" / str(id)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "node.json", "w") as f:
            f.write(out)

    @classmethod
    def load(cls, path: Path):
        data = json.load(open(path / "node.json", "r"))
        return cls(np.array(data["translation"], dtype=np.float64))

    def __eq__(self, other) -> bool:
        return (
            np.allclose(self.translation, other.translation)
            and self.virtual == other.virtual
        )

    def __hash__(self) -> int:
        return hash(self.translation.tostring())


class RealMapNode(MapNode):
    def __init__(
        self,
        image_l: ImageT,
        image_f: ImageT,
        image_r: ImageT,
        translation: Translation,
        rotation: Rotation,
        t: TimeT,  # From rospy.Time.to_secs
    ) -> None:
        super().__init__(translation)
        self.image_l = image_l
        self.image_f = image_f
        self.image_r = image_r
        self.t = t
        self.rotation = rotation

    @property
    def virtual(self):
        return False

    def serialize(self, path: Path, id: int):
        out = super().serialize(path, id)
        out_new = {"t": self.t, "rotation": self.rotation.tolist()}
        output_path = path / "nodes" / str(id)
        output_path.mkdir(parents=True, exist_ok=True)
        for fname, img in zip(
            ("left", "front", "right"), (self.image_l, self.image_f, self.image_r)
        ):
            fpath = output_path / f"{fname}.png"
            cv2.imwrite(str(fpath), img)
        out.update(out_new)
        return out

    def get_closest_image(self, other_node: MapNode):
        """
        TODO
        """
        return self.image_f

    @classmethod
    def load(cls, path: Path):
        data = json.load(open(path / "node.json", "r"))
        return cls(
            cv2.imread(str(path / "left.png")),
            cv2.imread(str(path / "front.png")),
            cv2.imread(str(path / "right.png")),
            np.array(data["translation"], dtype=np.float64),
            np.array(data["rotation"], dtype=np.float64),
            data["t"],
        )


class Visiblity(Enum):
    BIDIRECTIONAL = auto()
    ATOB = auto()
    BTOA = auto()
    NONE = auto()


class Traversability(Enum):
    TRAVERSABLE = auto()
    UNTRAVERSABLE = auto()
    DRIVEN = auto()
    UNKNOWN = auto()


class MapEdge:
    from topological_mapping.utils import project_node

    def __init__(
        self,
        nodeA: MapNode,
        nodeB: MapNode,
        traversability: Traversability,
        visibility: Optional[Visiblity] = None,
    ) -> None:
        self.nodeA = nodeA
        self.nodeB = nodeB
        if visibility is None:
            self.visibility = self.estabish_visiblity()
        else:
            self.visibility = visibility
        self.traversability = traversability
        if (
            self.traversability is Traversability.TRAVERSABLE
            or self.traversability is Traversability.UNTRAVERSABLE
        ):
            pass
            # Traversability can only be established between at least
            # one real node, and one-way visibility
            # assert (not nodeA.virtual) or (not nodeB.virtual)
            # assert self.visibility != Visiblity.NONE

    def estabish_visiblity(self):
        cam_name_a = None
        cam_name_b = None

        if isinstance(self.nodeA, RealMapNode):
            _, cam_name_a, _, _ = MapEdge.project_node(self.nodeA, self.nodeB, None)
        if isinstance(self.nodeB, RealMapNode):
            _, cam_name_b, _, _ = MapEdge.project_node(self.nodeB, self.nodeA, None)
        if cam_name_a is None and cam_name_b is not None:
            return Visiblity.BTOA
        elif cam_name_b is None and cam_name_a is not None:
            return Visiblity.ATOB
        elif cam_name_b is not None and cam_name_a is not None:
            return Visiblity.BIDIRECTIONAL
        else:
            return Visiblity.NONE

    def __eq__(self, __value: MapEdge) -> bool:
        if self.nodeA is __value.nodeA and self.nodeB is __value.nodeB:
            return True
        elif self.nodeA is __value.nodeB and self.nodeB is __value.nodeA:
            return True
        else:
            return False

    def serialize(self, map: Optional[TopologicalMap]):
        if map is None:
            idA = 0
            idB = 1
        else:
            idA = map.get_node_id(self.nodeA)
            idB = map.get_node_id(self.nodeB)
        try:
            return {
                "traversability": str(self.traversability),
                "nodeA": idA,
                "nodeB": idB,
                "visibility": str(self.visibility),
            }
        except ValueError:
            raise ValueError(f"Could save edge {self}")

    def __str__(self):
        return (
            f"Traversability: {str(self.traversability)} \n"
            + f"NodeA: {str(self.nodeA)} \n"
            + f"NodeB: {str(self.nodeB)} \n"
            + f"Visiblity: {str(self.visibility)} \n"
        )

    def save(self, map: Optional[TopologicalMap], path: Path, id: Optional[int] = None):
        if id is None:
            id = map.get_edge_id(self)
        output_path = path / "edges"
        out = json.dumps(self.serialize(map))
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / f"{id}.json", "w") as f:
            f.write(out)

    @classmethod
    def load(cls, path: Path, map: TopologicalMap):
        data = json.load(open(path, "r"))
        # print(f"{len(map.nodes)=}")
        # print(f"{data=}")
        return cls(
            map.nodes[data["nodeA"]],
            map.nodes[data["nodeB"]],
            eval(data["traversability"]),
            eval(data["visibility"]),
        )


class TopologicalMap:
    def __init__(
        self,
        resolution: float,
        grid_resolution: float,
        initial_node: RealMapNode,
        edge_label_callback: Callable[[MapNode, MapNode], Traversability],
        grid_size: float = 96,
        frontiers_disappear: bool = True,
        debug: bool = False,
    ) -> None:
        self.grid_resolution: float = grid_resolution
        self.resolution: float = resolution
        self.nodes: List[MapNode] = [initial_node]
        self.edges: List[MapEdge] = []
        self.current_node: RealMapNode = initial_node
        self.edge_label_callback = edge_label_callback
        self.frontiers_disappear = frontiers_disappear
        self.occupancy_map = OccupancyMap(
            -grid_size / 2.0, grid_size / 2.0, self.grid_resolution
        )
        self.occupancy_map.add_visit(initial_node.translation[:2])
        self.__regen_kdtree()
        self.debug = debug

    def validate_map(self):
        for node in self.nodes:
            assert len(set(node.reachable_neighbors)) == len(node.reachable_neighbors)
            assert len(set(node.unreachable_neighbors)) == len(
                node.unreachable_neighbors
            )
            for neighbor in node.reachable_neighbors:
                assert node in neighbor.reachable_neighbors
            for neighbor in node.unreachable_neighbors:
                assert node in neighbor.unreachable_neighbors
        assert self.twod_kdtree.data.shape[0] == len(self.nodes)
        assert self.current_node in self.nodes

    @classmethod
    def load(cls, path: Path):
        data = json.load(open(path / "map.json", "r"))
        map = None
        print("Loading nodes...")
        current_node = data["current_node"]
        print(path)
        for i, node_path in tqdm(
            enumerate(
                sorted(
                    (path / "nodes").glob("*"),
                    key=lambda folder_name: int(str(folder_name.stem)),
                )
            ),
            total=current_node,
        ):
            # if i > current_node:
            #     break
            node = MapNodeFactory(node_path)
            if map is None:
                if "grid_resolution" not in data.keys():
                    data["grid_resolution"] = 1.0
                map = cls(data["resolution"], data["grid_resolution"], node, None)
            else:
                # DO NOT go through add_node because it does the whole
                # K-nn search thing. We'll load the edges from the dataset.
                map.nodes.append(node)

            if i == current_node:
                map.current_node = node

        print("Loading edges...")
        for edge_path in tqdm(
            sorted(
                (path / "edges").glob("*.json"),
                key=lambda folder_name: int(str(folder_name.stem)),
            )
        ):
            try:
                edge = MapEdge.load(edge_path, map)
                # Rebuild the (un)reachable neighbors references
                map.add_edge(edge)
            except IndexError:
                break
        map.__regen_kdtree()
        return map

    def save(self, path):
        for n in self.nodes:
            n.save(self, path)
        for e in self.edges:
            e.save(self, path)
        data = {
            "resolution": self.resolution,
            "grid_resolution": self.grid_resolution,
            "current_node": self.get_node_id(self.current_node),
        }
        data = json.dumps(data)
        with open(path / "map.json", "w") as f:
            f.write(data)

    def get_edge_label(self, node, neighbor):
        if isinstance(node, RealMapNode):
            traversable = self.edge_label_callback(node, neighbor)
        elif isinstance(neighbor, RealMapNode):
            traversable = self.edge_label_callback(neighbor, node)
        else:
            traversable = Traversability.UNKNOWN
        edge = MapEdge(node, neighbor, traversable)
        return edge

    def __regen_kdtree(self):
        # FIXME incremental KDTree of some sort would be nice.
        # libnabo?
        # node_coords_array = np.array([x.translation for x in self.nodes])
        # self.kdtree = KDTree(node_coords_array)
        twod_node_coords_array = np.array([x.translation[:2] for x in self.nodes])
        self.twod_kdtree = KDTree(twod_node_coords_array)

    def add_edge(self, edge: MapEdge):
        self.edges.append(edge)
        if edge.traversability is not Traversability.UNTRAVERSABLE:
            edge.nodeA.reachable_neighbors.append(edge.nodeB)
            edge.nodeB.reachable_neighbors.append(edge.nodeA)
        else:
            edge.nodeA.unreachable_neighbors.append(edge.nodeB)
            edge.nodeB.unreachable_neighbors.append(edge.nodeA)

    def get_frontiers(self):
        return [node for node in self.nodes if not isinstance(node, RealMapNode)]

    def get_reachable_frontiers(self):
        fs = [node for node in self.get_frontiers() if node.reachable_neighbors]
        if len(fs) == 0:
            rospy.logwarn("DANGER")
        return fs

    def add_node(self, node: MapNode):
        # for other_node in self.nodes:
        #     if isinstance(other_node, RealMapNode):
        #         # Don't add node if within same cell (TODO better)
        #         if self.occupancy_map.get_buffer_coordinates(
        #             other_node.translation[:2]
        #         ) == self.occupancy_map.get_buffer_coordinates(node.translation[:2]):
        #             self.current_node = other_node
        #             return
        if self.debug:
            self.validate_map()

        self.nodes.append(node)

        if self.debug:
            self.validate_map()

        if isinstance(node, RealMapNode):
            self.occupancy_map.add_visit(node.translation[:2])
            edge = MapEdge(self.current_node, node, Traversability.DRIVEN)
            self.add_edge(edge)
            if self.frontiers_disappear:
                for other_node in self.nodes:
                    if not isinstance(other_node, RealMapNode):
                        # Remove frontier if real node fall within its cell
                        if self.occupancy_map.get_buffer_coordinates(
                            other_node.translation[:2]
                        ) == self.occupancy_map.get_buffer_coordinates(
                            node.translation[:2]
                        ):
                            self.remove_node(other_node, regen_kdtree=False)
        self.__regen_kdtree()

        if self.debug:
            self.validate_map()

        for neighbor in self.query_neighbors(node):
            if (
                neighbor is self.current_node and isinstance(node, RealMapNode)
            ) or neighbor is node:
                # Already added the edge, do nothing
                # Because we drove on that edge no need to run
                # traversability analysis.
                continue
            edge = self.get_edge_label(node, neighbor)
            self.add_edge(edge)

        if isinstance(node, RealMapNode):
            self.current_node = node

        if self.debug:
            self.validate_map()

    def query_neighbors(self, node: MapNode) -> List[MapNode]:
        # FIXME incremental KDTree of some sort would be nice.
        # libnabo?
        neighbors = self.twod_kdtree.query_ball_point(
            node.translation[:2], self.resolution
        )
        return [self.nodes[i] for i in neighbors]

    def remove_node(self, node: MapNode, regen_kdtree=True):
        self.nodes.remove(node)
        edges_to_remove = []
        for edge in self.edges:
            if edge.nodeA is node or edge.nodeB is node:
                edges_to_remove.append(edge)
        for edge in edges_to_remove:
            self.edges.remove(edge)
        for other_node in node.unreachable_neighbors:
            other_node.unreachable_neighbors.remove(node)
        for other_node in node.reachable_neighbors:
            other_node.reachable_neighbors.remove(node)
        if regen_kdtree:
            self.__regen_kdtree()
            if self.debug:
                self.validate_map()
        for edge in self.edges:
            if not edge.nodeA in self.nodes and edge.nodeB in self.nodes:
                assert False

    def get_edge_by_nodes(self, nodeA: MapNode, nodeB: MapNode):
        for e in self.edges:
            if e.nodeA is nodeA and e.nodeB is nodeB:
                return e
            elif e.nodeB is nodeA and e.nodeA is nodeB:
                return e

        # If not identical node, check for identical translation
        # this can be helpful in the topological simulator where we're simulating frontiers
        for e in self.edges:
            if np.allclose(e.nodeA.translation, nodeA.translation) and np.allclose(
                e.nodeB.translation, nodeB.translation
            ):
                return e
            elif np.allclose(e.nodeB.translation, nodeA.translation) and np.allclose(
                e.nodeA.translation, nodeB.translation
            ):
                return e

    def get_node_id(self, node):
        for i, n in enumerate(self.nodes):
            if n is node:
                return i
        raise ValueError(f"Node {node} not found")

    def get_edge_id(self, edge):
        for i, e in enumerate(self.edges):
            if e is edge:
                return i
        raise ValueError(f"Edge {edge} not found")

    def relabel_edge(
        self,
        edge: Optional[MapEdge],
        nodeA: Optional[MapNode],
        nodeB: Optional[MapNode],
        new_label: Traversability,
    ) -> None:
        if edge is None:
            for e in self.edges:
                if (e.nodeA is nodeA and e.nodeB is nodeB) or (
                    e.nodeB is nodeA and e.nodeA is nodeB
                ):
                    edge = e
                    break
            if edge is None:
                raise ValueError("Trying to relabel an edge that doesn't exist")

        # Remove nodes pointers
        if edge.traversability in [
            Traversability.DRIVEN,
            Traversability.TRAVERSABLE,
            Traversability.UNKNOWN,
        ]:
            nodeA.reachable_neighbors.remove(nodeB)
            nodeB.reachable_neighbors.remove(nodeA)
        else:
            nodeA.unreachable_neighbors.remove(nodeB)
            nodeB.unreachable_neighbors.remove(nodeA)

        # Relabel and re-add nodes pointers to appropriate list
        edge.traversability = new_label
        if edge.traversability in [
            Traversability.DRIVEN,
            Traversability.TRAVERSABLE,
            Traversability.UNKNOWN,
        ]:
            nodeA.reachable_neighbors.append(nodeB)
            nodeB.reachable_neighbors.append(nodeA)
        else:
            nodeA.unreachable_neighbors.append(nodeB)
            nodeB.unreachable_neighbors.append(nodeA)

    def grow_map(self):
        self.occupancy_map.compute_frontiers()
        virtual_nodes_coords = self.occupancy_map.get_frontiers()

        for i in range(virtual_nodes_coords.shape[0]):
            neighbors = self.twod_kdtree.query_ball_point(
                virtual_nodes_coords[i], 2 * self.grid_resolution
            )
            self.twod_kdtree.query(virtual_nodes_coords[i], 4 * self.resolution)
            neighbors = [self.nodes[i] for i in neighbors]
            if any(
                [
                    np.linalg.norm(n.translation[:2] - virtual_nodes_coords[i])
                    < 0.5 * self.grid_resolution
                    for n in neighbors
                ]
            ):
                continue
            neighbors = [n for n in neighbors if isinstance(n, RealMapNode)]
            neighbors_dist = [
                np.linalg.norm(virtual_nodes_coords[i] - n.translation[:2])
                for n in neighbors
            ]
            if len(neighbors) == 0:
                height = 0
            else:
                closest_neighbor = neighbors[np.argmin(neighbors_dist)]
                height = closest_neighbor.translation[2]
            coords = np.array(
                [
                    virtual_nodes_coords[i, 0],
                    virtual_nodes_coords[i, 1],
                    height,
                ]
            )
            node = MapNode(coords)
            self.add_node(node)
            self.__regen_kdtree()
