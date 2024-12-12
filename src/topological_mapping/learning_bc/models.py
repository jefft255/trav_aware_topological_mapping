from topological_mapping.learning_trav.models import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid

import matplotlib.pyplot as plt
from contextlib import nullcontext

import numpy as np


class DINOv2Enc(nn.Module):
    def __init__(self, version):
        super().__init__()
        assert version == "s"
        self.version = version
        self.dino = torch.hub.load("facebookresearch/dinov2", f"dinov2_vit{version}14")
        self.last_feature_dim = 10
        self.dim_reduction = nn.Sequential(
            nn.Conv2d(384, 100, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(100, 100, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(100, self.last_feature_dim, 1, padding=0),
        )
        self.output_dim = ((224 // 14) ** 2) * self.last_feature_dim

    def forward(self, img):
        with torch.no_grad():
            x = self.dino.forward_features(img)
        x = x["x_prenorm"][:, :-1]
        x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], 384, 16, 16)
        x = self.dim_reduction(x)
        x = x.reshape(x.shape[0], self.output_dim)
        return x


class DINOv2BCNet(nn.Module):
    def __init__(
        self,
        mu,
        std,
        version="l",
        hist_size=1,
        pred_horizon=1,
        only_front=False,
        past_actions=False,
        velocities=False,
        hidden_layers=2,
        hidden_dim=1024,
        residual=False,
        gelu=False,
    ) -> None:
        super().__init__()

        self.mu = mu
        self.std = std

        self.only_front = only_front
        self.version = version
        self.goal_dim = 2
        self.control_dim = 2
        self.hist_size = hist_size
        self.pred_horizon = pred_horizon
        self.velocities = velocities
        self.past_actions = past_actions

        if self.only_front:
            self.n_images = self.hist_size
        else:
            self.n_images = 3 * self.hist_size

        self.img_enc = self.get_img_enc(version)
        input_dim = self.goal_dim + self.n_images * self.img_enc_dim

        if self.past_actions:
            input_dim += self.hist_size * self.control_dim

        if self.velocities:
            input_dim += self.hist_size * 12

        print(f"Input dim: {input_dim}")

        self.fuser = MLP(
            input_dim,
            self.pred_horizon * self.control_dim,
            hidden_dim,
            hidden_layers,
            residual,
            gelu,
        )

    @property
    def img_enc_dim(self):
        return self.img_enc.output_dim

    def get_img_enc(self, version):
        return DINOv2Enc(version)

    def forward(
        self, images_l, images_f, images_r, vels, actions, goal, images_enc=None
    ):
        goal = goal[:, :2]
        B = actions.shape[0]
        if images_enc is None:
            # Stack in the "history" dimension
            if self.only_front:
                H = images_f.shape[1]
                assert H == self.hist_size
                images = torch.reshape(images_f, (B * H, *images_f.shape[2:]))
            else:
                images = torch.concatenate([images_l, images_f, images_r], dim=1)
                H = images.shape[1]
                assert H == 3 * self.hist_size
                images = torch.reshape(images, (B * H, *images.shape[2:]))

            images_enc = self.img_enc(images)
            images_enc = torch.reshape(images_enc, (B, H, self.img_enc_dim))
        else:
            assert not self.only_front
            H = 3 * self.hist_size

        images_enc = torch.reshape(images_enc, (B, H * self.img_enc_dim))
        model_in_list = [images_enc, goal]
        if self.past_actions:
            actions = torch.reshape(actions, (B, self.hist_size * self.control_dim))
            model_in_list.append(actions)
        if self.velocities:
            vels = vels.reshape(B, self.hist_size * 12)
            model_in_list.append(vels)
        model_in = torch.concatenate(model_in_list, dim=-1)
        # out = F.tanh(self.fuser(model_in))
        out = self.fuser(model_in)
        actions = out.reshape(B, self.pred_horizon, self.control_dim)
        return actions


class ResNetBCNet(DINOv2BCNet):
    def __init__(
        self,
        mu,
        std,
        # mu_goal,
        # std_goal,
        # mu_img,
        # std_img,
        version="l",
        hist_size=1,
        pred_horizon=1,
        only_front=False,
        past_actions=False,
        velocities=False,
        hidden_layers=2,
        hidden_dim=1024,
        residual=False,
        gelu=False,
    ) -> None:
        super().__init__(
            mu,
            std,
            # mu_goal,
            # std_goal,
            # mu_img,
            # std_img,
            version,
            hist_size,
            pred_horizon,
            only_front,
            past_actions,
            velocities,
            hidden_layers,
            hidden_dim,
            residual,
            gelu,
        )

    @property
    def is_encoder_pretrained(self):
        return False

    def get_img_enc(self, version):
        try:
            resnet = torch.hub.load(
                "pytorch/vision:v0.10.0", f"resnet{version}", weights=None
            )
        except TypeError:
            resnet = torch.hub.load(
                "pytorch/vision:v0.10.0", f"resnet{version}", pretrained=False
            )
        compress = nn.Linear(1000, self.img_enc_dim)
        return nn.Sequential(resnet, compress)

    @property
    def img_enc_dim(self):
        return 512
