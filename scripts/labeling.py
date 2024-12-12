from typing import Any, Optional, List
from typing_extensions import Literal
import pickle
from time import time
import sys
import json
from pathlib import Path
from tqdm import tqdm

from topological_mapping.topological_map import (
    RealMapNode,
    MapNode,
    MapNodeFactory,
    TopologicalMap,
    Traversability,
)
from topological_mapping.utils import project_node

import cv2

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")


IMG_WIDTH = 300
IMG_HEIGTH = 225


def draw_context(nodes, nodeA, nodeB):
    dpi = 50
    size = 2 * IMG_HEIGTH
    fig = plt.figure(
        figsize=(float(size) / float(dpi), float(size) / float(dpi)), dpi=dpi
    )
    ax = fig.gca()
    ax.scatter(
        [n.translation[0] for n in nodes], [n.translation[1] for n in nodes], s=2
    )
    for n in (nodeA, nodeB):
        if hasattr(n, "rotation"):
            for i, c in zip(range(3), ["red", "yellow", "blue"]):
                dir = n.rotation[:, i]
                x = [n.translation[0], n.translation[0] + dir[0]]
                y = [n.translation[1], n.translation[1] + dir[1]]
                ax.plot(x, y, color=c, lw=5, alpha=1.0)
    ax.scatter([nodeA.translation[0]], [nodeA.translation[1]], s=800, color="yellow")
    ax.scatter([nodeB.translation[0]], [nodeB.translation[1]], s=800, color="blue")
    margin = 1

    y_min = min(nodeA.translation[1], nodeB.translation[1]) - margin
    y_max = max(nodeA.translation[1], nodeB.translation[1]) + margin
    x_min = min(nodeA.translation[0], nodeB.translation[0]) - margin
    x_max = max(nodeA.translation[0], nodeB.translation[0]) + margin
    if (x_max - x_min) > (y_max - y_min):
        y_max = y_min + x_max - x_min
    else:
        x_max = x_min + y_max - y_min
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (size, size, 3)
    plt.close("all")
    return buf


def process_image(img):
    return cv2.resize(img, (IMG_WIDTH, IMG_HEIGTH))


def draw_two_nodes(nodes, nodeA, nodeB):
    images = [[], []]
    visible = False
    for i, n in enumerate([nodeA, nodeB]):
        if hasattr(n, "image_f"):
            # Real node; do visibility analysis and projection
            if i == 0:
                img, cam, _, _ = project_node(nodeA, nodeB, "blue")
            else:
                img, cam, _, _ = project_node(nodeB, nodeA, "yellow")
            visible = visible or (cam is not None)
            cam = {"left": 0, "front": 1, "right": 2, None: -1}[cam]
            for j in range(3):
                if j == cam:
                    images[i].append(process_image(img))
                else:
                    images[i].append(
                        process_image([n.image_l, n.image_f, n.image_r][j])
                    )
        else:
            # Virtual node without images; display black
            for _ in range(3):
                images[i].append(process_image(np.zeros((100, 100, 3), dtype=np.uint8)))
    if not visible:
        return False
    images = [cv2.hconcat(row) for row in images]
    images = cv2.vconcat(images)

    context_plot = draw_context(nodes, nodeA, nodeB)
    context_plot = cv2.cvtColor(context_plot, cv2.COLOR_RGB2BGR)
    final = cv2.hconcat([images, context_plot])
    cv2.imshow("Labeling", final)
    return True


def autolabel_temporal_neighbors(nodeA, nodeB):
    # TODO won't work perfectly. Need to search path and see if it's "direct"
    return np.abs(nodeA.t - nodeB.t) < 10.0


def callback_labeling(nodes, nodeA, nodeB):
    # temporal_neighbor = autolabel_temporal_neighbors(nodeA, nodeB)
    # if temporal_neighbor:
    #     return Traversability.DRIVEN
    # else:
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
callback_labeling_closure = lambda a, b: callback_labeling(nodes, a, b)
if (data_path / "map.json").exists():
    # Labeling progress saved, resume
    tmap = TopologicalMap.load(data_path)
    tmap.frontiers_disappear = False
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
    nodes.append(MapNodeFactory(n_path))

print(f"Total nodes: {len(nodes)}")
for i, node in tqdm(enumerate(nodes), total=len(nodes)):
    if i < current_node_id:
        continue
    if tmap is None:
        tmap = TopologicalMap(4.0, 2.0, node, callback_labeling_closure, 100, False)
    else:
        tmap.add_node(node)
        print("====== Growing map =======")
        tmap.grow_map()
        print("====== Done growing map =======")
    if i % 10 == 0:
        print("Saving...")
        tmap.save(Path(str(data_path) + "_labeled"))
tmap.save(data_path)
