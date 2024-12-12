import sys
import os
import json
from pathlib import Path
from tqdm import tqdm

from topological_mapping.topological_map import (
    RealMapNode,
    TopologicalMap,
    Traversability,
)

import cv2

import numpy as np
import numpy.typing as npt

sys.path.append(os.path.dirname(__file__))
from labeling import draw_two_nodes, autolabel_temporal_neighbors

IMG_WIDTH = 300
IMG_HEIGTH = 225


def callback_labeling(nodes, model, nodeA, nodeB):
    temporal_neighbor = autolabel_temporal_neighbors(nodeA, nodeB)
    if temporal_neighbor:
        return Traversability.DRIVEN

    model_out = model

    visible = draw_two_nodes(nodes, nodeA, nodeB)
    if not visible:
        print("Skipping labeling between non-co-visible nodes.")
        return Traversability.UNKNOWN
    while True:
        key = cv2.waitKey(0)
        # Key is t
        if key == 116:
            cv2.destroyAllWindows()
            return Traversability.TRAVERSABLE
        # Key is r
        if key == 114:
            cv2.destroyAllWindows()
            return Traversability.UNTRAVERSABLE


data_path = Path(sys.argv[1])
model_path = Path(sys.argv[2])

callback_labeling_closure = lambda a, b: callback_labeling(nodes, a, b)
if (data_path / "map.json").exists():
    # Labeling progress saved, resume
    tmap = TopologicalMap.load(data_path)
    tmap.edge_label_callback = callback_labeling_closure
    current_node_id = json.load(open(data_path / "map.json", "r"))["current_node"] + 1
    print(f"---- Resuming lababeling at node {current_node_id} ----")
    print(f"Nodes: {len(tmap.nodes)}")
    print(f"Edges: {len(tmap.edges)}")
    print(
        f"T Edges: {len([e for e in tmap.edges if e.traversability is Traversability.TRAVERSABLE])}"
    )
    print(
        f"UNT Edges: {len([e for e in tmap.edges if e.traversability is Traversability.UNTRAVERSABLE])}"
    )
    print(
        f"D Edges: {len([e for e in tmap.edges if e.traversability is Traversability.DRIVEN])}"
    )
    print(
        f"UNKNOWN Edges: {len([e for e in tmap.edges if e.traversability is Traversability.UNKNOWN])}"
    )
else:
    current_node_id = 0
    tmap = None

nodes = []
for n_path in sorted(
    (data_path / "nodes").glob("*"),
    key=lambda folder_name: int(str(folder_name.stem)),
):
    nodes.append(RealMapNode.load(n_path))

for i, node in tqdm(enumerate(nodes)):
    if i < current_node_id:
        continue
    if tmap is None:
        tmap = TopologicalMap(3.0, node, callback_labeling_closure)
    else:
        tmap.add_node(node)
    if i % 50 == 0:
        print("Saving...")
        tmap.save(data_path)
tmap.save(data_path)
