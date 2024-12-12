import torch
from topological_mapping.learning_bc.dataset import BCDataset
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.utils import make_grid
from topological_mapping.learning_bc.models import DINOv2BCNetDiscrete
import torch.nn.functional as F
import sys
from math import sin, cos
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf
import socket
import numpy as np

classification_criterion = torch.nn.CrossEntropyLoss().cuda()


def validation(val_dataset, model, logger, epoch, cfg, device, mu, std):
    with torch.no_grad():
        avg_loss = 0.0
        val_dataloader = DataLoader(val_dataset, cfg.batch_size, True)
        for batch in tqdm(val_dataloader):
            imgs_l, imgs_f, imgs_r, goal, cmd = batch
            ###cmd = (cmd - mu) / std
            ######################################### Ground truth preparation for classification ######################
            class_labels = model.discretize_cmd(cmd)
            class_labels = torch.tensor(class_labels, dtype=torch.long).to(device)
            ##########################################################################################

            imgs_l = imgs_l.to(device)
            imgs_f = imgs_f.to(device)
            imgs_r = imgs_r.to(device)
            goal = goal.to(device)
            cmd = cmd.to(device)
            out = model(imgs_l, imgs_f, imgs_r, goal)
            loss = classification_criterion(out, class_labels)
            avg_loss += loss.item()

        print(avg_loss / len(val_dataloader))
        logger.add_scalar("Loss/val", avg_loss / len(val_dataloader), epoch)


def validation_goal(val_dataset, model, logger, epoch, cfg, device, mu, std):  # TODO
    with torch.no_grad():
        val_dataloader = DataLoader(val_dataset, cfg.batch_size, True)
        i = 0
        n_thingy = 3
        fig, axs = plt.subplots(n_thingy * 4, 4, dpi=300)
        fig.set_size_inches(4, 12)
        for batch in tqdm(val_dataloader):
            imgs_l, imgs_f, imgs_r, goal, cmd = batch
            imgs_l = imgs_l[:4].to(device)
            imgs_f = imgs_f[:4].to(device)
            imgs_r = imgs_r[:4].to(device)
            goal = goal[:4]
            # goal[0] = torch.tensor([1, 0, 0])
            # goal[1] = torch.tensor([-1, 0, 0])
            # goal[2] = torch.tensor([0, -1, 0])
            # goal[3] = torch.tensor([0, 1, 0])
            goal = goal.to(device)
            out = model(imgs_l, imgs_f, imgs_r, goal)
            out = std * out.cpu() + mu
            cmd = std * cmd.cpu() + mu
            for j in range(4):
                axs[4 * i + j, 3].arrow(
                    0,
                    0,
                    out[j, 0] * sin(out[j, 1]),
                    out[j, 0] * cos(out[j, 1]),
                    color="yellow",
                    label="output",
                )
                axs[4 * i + j, 3].arrow(
                    0, 0, goal[j, 1].cpu(), goal[j, 0].cpu(), color="red", label="goal"
                )
                axs[4 * i + j, 3].arrow(
                    0, 0, cmd[j, 1].cpu(), cmd[j, 0].cpu(), color="blue", label="gt"
                )
                axs[4 * i + j, 3].set_ylim(-1, 1)
                axs[4 * i + j, 3].set_xlim(-1, 1)
                axs[4 * i + j, 0].imshow(imgs_l[j, -1].cpu().permute(1, 2, 0))
                axs[4 * i + j, 1].imshow(imgs_f[j, -1].cpu().permute(1, 2, 0))
                axs[4 * i + j, 2].imshow(imgs_r[j, -1].cpu().permute(1, 2, 0))
            i += 1
            if i == n_thingy:
                break
        for i in range(n_thingy * 4):
            for j in range(4):
                axs[i, j].axis("off")
        fig.tight_layout()
        logger.add_figure("GoalDebug", fig, epoch)


if socket.gethostname() == "raza":
    config_path = "/home/adaptation/jft/catkin_ws/src/topological_mapping/config"
else:
    config_path = "/home/barbados/JF_data/src/topological_mapping/config"


@hydra.main(
    version_base=None,
    config_path=config_path,
    config_name="train_bc_discrete",
)
def my_app(cfg: DictConfig) -> None:
    # torch.backends.cuda.matmul.allow_tf32 = (
    #     True  # Enable/disable TF32 for matrix multiplications
    # )
    # torch.backends.cudnn.allow_tf32 = True  # Enable/disable TF32 for convolutions
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger = SummaryWriter(output_dir)
    data_path = Path(cfg.data_path)
    dataset = BCDataset(cfg.hist_size, data_path, cfg.mirror_trajs)
    train_dataset, val_dataset = random_split(dataset, [0.95, 0.05])
    if cfg.tiny_dataset:
        train_dataset, _ = random_split(train_dataset, [0.05, 0.95])
    mu, std = dataset.get_mean_std_commands()
    mu, std = torch.tensor(mu), torch.tensor(std)
    print(f"Mu {mu}")
    print(f"Std {std}")
    test_dataset = None  # TODO different map!
    train_dataloader = DataLoader(
        train_dataset,
        cfg.batch_size,
        True,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        # pin_memory=True,
    )
    device = torch.device("cuda")
    model = DINOv2BCNetDiscrete(
        mu,
        std,
        version=cfg.version,
        no_image=cfg.no_image,
        hist_size=cfg.hist_size,
        only_front=cfg.only_front,
        hidden_layers=cfg.hidden_layers,
        hidden_dim=cfg.hidden_dim,
        linear_vel_dim=cfg.linear_vel_dim,
        angular_vel_dim=cfg.angular_vel_dim,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)

    for epoch in range(cfg.n_epochs):
        # if epoch==16:
        #     optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate*0.1)
        print(f"===== Epoch {epoch} =====")
        avg_loss = 0.0
        for batch in tqdm(train_dataloader):
            imgs_l, imgs_f, imgs_r, goal, cmd = batch
            ######################################### Ground truth preparation for classification ######################
            class_labels = model.discretize_cmd(cmd)
            class_labels = torch.tensor(class_labels, dtype=torch.long).to(device)
            ##########################################################################################
            # cmd = (cmd - mu) / std
            imgs_l = imgs_l.to(device)
            imgs_f = imgs_f.to(device)
            imgs_r = imgs_r.to(device)
            goal = goal.to(device)
            cmd = cmd.to(device)
            optimizer.zero_grad()
            out = model(imgs_l, imgs_f, imgs_r, goal)
            # The output would be a vector of 12 size
            loss = classification_criterion(out, class_labels)
            avg_loss += loss.item()  # TODO .item attribute needs to be changed I think
            loss.backward()
            optimizer.step()

        logger.add_scalar("Loss/train", avg_loss / len(train_dataloader), epoch)
        if epoch % 1 == 0:
            print("===== Running goal validation ======")
            #####validation_goal(val_dataset, model, logger, epoch, cfg, device, mu, std)
            # print("===== Running train eval ======")
            # validation(train_dataset, model)
            print("===== Running validation ======")
            validation(val_dataset, model, logger, epoch, cfg, device, mu, std)
            torch.save(model, open(output_dir / "model_bc.pth", "wb"))


if __name__ == "__main__":
    my_app()
