import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18
from contextlib import nullcontext


class MLP(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim, n_layers, residual=False, gelu=False
    ):
        super().__init__()
        self.residual = residual
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(n_layers)]
        )
        self.f = F.gelu if gelu else F.relu
        self.last_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if not hasattr(self, "f"):
            self.f = F.relu
        if not hasattr(self, "residual"):
            self.residual = False
        x = self.f(self.first_layer(x))
        for l in self.hidden_layers:
            if not self.residual:
                x = self.f(l(x))
            else:
                x = self.f(l(x) + x)
        return self.last_layer(x)


class TraversabilityAnalyser(nn.Module):
    def __init__(self, hidden_dim, n_layers, one_hot) -> None:
        super().__init__()
        self.image_encoder = self.get_image_encoder()
        # self.image_encoder.eval()
        relative_pose_dim = 3 if not one_hot else (224 // 14) ** 2
        # self.fuser = MLP(
        #     # self.image_encoder_dim() + relative_pose_dim, 2, hidden_dim, n_layers
        #     self.image_encoder_dim(),
        #     2,
        #     hidden_dim,
        #     n_layers,
        # )
        self.fuser = MLP(
            # self.image_encoder_dim() + relative_pose_dim, 2, hidden_dim, n_layers
            self.image_encoder_dim(),
            2,
            hidden_dim,
            n_layers,
        )
        # self.dim_reduction = nn.Sequential(
        #     nn.Conv2d(384, 100, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(100, 100, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(100, 10, 3, padding=1),
        # )
        self.dim_reduction = nn.Sequential(
            nn.Conv2d(384, 100, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(100, 100, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(100, 10, 1, padding=0),
        )
        self.pretrained_dimred = False

    def get_image_encoder(self):
        raise NotImplementedError("")

    def image_encoder_dim(self):
        raise NotImplementedError("")

    @property
    def is_encoder_pretrained(self):
        raise NotImplementedError("")

    def forward(self, imgA, relative_pose):
        # context = torch.no_grad() if self.is_encoder_pretrained else nullcontext()
        with torch.no_grad():
            encA = self.image_encoder.forward_features(imgA)
            # print([str(k) for k in encA.keys()])
            encA = encA["x_prenorm"][:, :-1]
        if not hasattr(self, "pretrained_dimred"):
            self.pretrained_dimred = False
        context = torch.no_grad() if self.pretrained_dimred else nullcontext()
        with context:
            x = encA.permute(0, 2, 1)
            # print(f"{x.shape}")
            x = x.reshape(x.shape[0], 384, 16, 16)
            x = self.dim_reduction(x)
            # x = torch.concatenate([x, relative_pose], dim=1)
            x = x.reshape(x.shape[0], -1)
            x = torch.concatenate([x, relative_pose], dim=-1)
        return self.fuser(x)

    def multi_goal_forward(self, imgA, relative_poses):
        B = relative_poses.shape[0]
        with torch.no_grad():
            encA = self.image_encoder.forward_features(imgA.unsqueeze(dim=0))
            encA = encA["x_norm_patchtokens"]
        encA = encA.expand(B, -1, -1)
        x = encA.permute(0, 2, 1)
        x = x.reshape(x.shape[0], 384, 16, 16)
        x = self.dim_reduction(x)
        # x = torch.concatenate([x, relative_pose], dim=1)
        x = x.reshape(x.shape[0], -1)
        x = torch.concatenate([x, relative_poses], dim=-1)
        return self.fuser(x)


class ResNetTraversabilityAnalyser(TraversabilityAnalyser):
    def __init__(self, hidden_dim, n_layers, version, one_hot) -> None:
        self.version = version
        super().__init__(hidden_dim, n_layers, one_hot)

    def get_image_encoder(self):
        return resnet18()

    def image_encoder_dim(self):
        return 1000

    @property
    def is_encoder_pretrained(self):
        return False


class DINOv2TraversabilityAnalyser(TraversabilityAnalyser):
    def __init__(self, hidden_dim, n_layers, version, one_hot) -> None:
        self.dims = {"s": 384, "b": 768, "l": 1024, "g": 1536}
        self.version = version
        super().__init__(hidden_dim, n_layers, one_hot)

    def get_image_encoder(self):
        return torch.hub.load("facebookresearch/dinov2", f"dinov2_vit{self.version}14")

    def image_encoder_dim(self):
        # return self.dims[self.version]
        return 2816
        # return 98560

    @property
    def is_encoder_pretrained(self):
        return True
