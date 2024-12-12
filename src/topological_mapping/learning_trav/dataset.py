import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Any
from pathlib import Path
import cv2
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Compose,
    ColorJitter,
    GaussianBlur,
)

from topological_mapping.topological_map import (
    TopologicalMap,
    Traversability,
    Visiblity,
    MapEdge,
    MapNodeFactory,
)
from topological_mapping.utils import project_node
import json


class Unnormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class TraversabilityDataset(Dataset):
    def __init__(
        self, map_paths: List[Path], ss_paths: List[Path], size=(224, 224), train=True
    ) -> None:
        super().__init__()
        self.maps = [TopologicalMap.load(p) for p in map_paths]
        # Unidirectional for now. So we're going to double some edges.
        self.edges = []
        self.size = size
        self.train = train
        self.position_noise_std = 0.1

        for map in self.maps:
            for e in map.edges:
                self.process_edge(e)
        #
        for ss_path in ss_paths:
            ss_path = ss_path / "traversability_dataset/"
            for edge_path in ss_path.glob("*"):
                nodeA = MapNodeFactory(edge_path / "nodes" / "0")
                nodeB = MapNodeFactory(edge_path / "nodes" / "1")
                edge_data = json.load(open(edge_path / "edges" / "0.json", "r"))
                e = MapEdge(
                    nodeA,
                    nodeB,
                    eval(edge_data["traversability"]),
                    eval(edge_data["visibility"]),
                )
                self.process_edge(e)

        if self.train:
            self.transform = Compose(
                [
                    ToTensor(),
                    ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2),
                    GaussianBlur(5, sigma=(0.001, 2.0)),
                    # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transform = Compose(
                [
                    ToTensor(),
                    # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        self.untransform = Unnormalize(
            # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            mean=(0, 0, 0),
            std=(1, 1, 1),
        )

        n_trav = len([e for e in self.edges if e[2] == Traversability.TRAVERSABLE])
        n_untrav = len([e for e in self.edges if e[2] == Traversability.UNTRAVERSABLE])
        print(
            "Dataset balance: ",
            n_trav,
            n_untrav,
        )
        self.weights = [
            (
                1.0 / float(n_untrav)
                if e[2] == Traversability.UNTRAVERSABLE
                else 1.0 / float(n_trav)
            )
            for e in self.edges
        ]

    def pixel_to_one_hot(self, pixel, depth, img):
        dino_patch_size = 14
        dino_res = self.size[0] // dino_patch_size
        one_hot = np.zeros((dino_res, dino_res), dtype=np.float32)
        scale_x = float(dino_res) / float(img.shape[1])
        scale_y = float(dino_res) / float(img.shape[0])
        coord_x = int(scale_x * pixel[0])
        coord_y = int(scale_y * pixel[1])
        one_hot[coord_x, coord_y] = depth
        return one_hot.reshape((dino_res**2))

    def process_edge(self, e: MapEdge):
        # CAREFUL for now ignore "driven node". We have enough positive data as it is.
        if e.traversability not in [
            Traversability.TRAVERSABLE,
            Traversability.UNTRAVERSABLE,
        ]:
            return
        # TODO only translation should be useful
        n_noisy_sample = 10 if self.train else 1

        for i in range(n_noisy_sample):
            if e.visibility in [Visiblity.ATOB, Visiblity.BIDIRECTIONAL]:
                self.process_unidirectional_endge(
                    e.nodeA, e.nodeB, e.traversability, i != 0
                )
            if e.visibility in [Visiblity.BTOA, Visiblity.BIDIRECTIONAL]:
                self.process_unidirectional_endge(
                    e.nodeB, e.nodeA, e.traversability, i != 0
                )

    def process_unidirectional_endge(self, nodeA, nodeB, traversability, add_noise):
        if add_noise:
            oldA_translation = nodeA.translation.copy()
            oldB_translation = nodeB.translation.copy()
            nodeA.translation += np.random.randn(3) * self.position_noise_std
            nodeB.translation += np.random.randn(3) * self.position_noise_std
        imgA, _, TA, pixel = project_node(nodeA, nodeB, None)
        if TA is None:
            # That means that adding noise made the other node invisible
            nodeA.translation = oldA_translation
            nodeB.translation = oldB_translation
            return
        # _point means there's the goal projected in the image, visible.
        imgA_point, _, _, _ = project_node(nodeA, nodeB, "blue")
        one_hot = self.pixel_to_one_hot(pixel, TA[2], imgA)
        imgA = cv2.resize(imgA, self.size, interpolation=cv2.INTER_AREA)
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
        imgA_point = cv2.resize(imgA_point, self.size, interpolation=cv2.INTER_AREA)
        imgA_point = cv2.cvtColor(imgA_point, cv2.COLOR_BGR2RGB)
        self.edges.append(
            (
                imgA,
                TA,
                traversability,
                imgA_point,
                one_hot,
            )
        )
        if add_noise:
            nodeA.translation = oldA_translation
            nodeB.translation = oldB_translation

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, index) -> Any:
        img, T, trav, img_point, pixel_onehot = self.edges[index]
        # CAREFUL FIXME hack for now
        trav = [1, 0] if trav == Traversability.TRAVERSABLE else [0, 1]
        return (
            self.transform(img),
            T.astype(np.float32),
            torch.tensor(trav, dtype=torch.float32),
            self.transform(img_point),
            pixel_onehot,
        )
