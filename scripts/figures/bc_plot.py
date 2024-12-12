from pathlib import Path
from tqdm import tqdm
import sys

import torch
from torch.utils.data import DataLoader

from topological_mapping.learning_bc.dataset import BCDataset
from topological_mapping.learning_bc.models import DINOv2BCNet, ResNetBCNet
from topological_mapping.learning_bc.validation import (
    validation,
)

import hydra
from omegaconf import DictConfig
import time
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns

mpl.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["figure.facecolor"] = "#f7f7f2"
plt.rcParams["axes.facecolor"] = "#f7f7f2"
plt.rcParams["axes.edgecolor"] = "#cecdca"
plt.rcParams["grid.color"] = "#cecdca"


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def validation(val_dataset, model, cfg, device, mu, std, ax_vel, ax_ang, plot_label):
    with torch.no_grad():
        model.eval()
        avg_loss = 0.0
        val_dataloader = DataLoader(val_dataset, cfg.batch_size, True)
        loss_per_traj = []
        for batch in tqdm(val_dataloader):
            imgs_l, imgs_f, imgs_r, vels, past_cmds, goal, cmd, info = batch
            past_cmds = (past_cmds - mu) / std
            cmd = (cmd - mu) / std
            imgs_l = imgs_l.to(device)
            imgs_f = imgs_f.to(device)
            imgs_r = imgs_r.to(device)
            vels = vels.to(device)
            goal = goal.to(device)
            cmd = cmd.to(device)
            past_cmds = past_cmds.to(device)
            out = model(imgs_l, imgs_f, imgs_r, vels, past_cmds, goal)

            with torch.no_grad():
                for t in range(0, cfg.pred_horizon, 8):
                    cmd = cmd.cpu() * std + mu
                    out = out.cpu() * std + mu
                    data = {
                        "Trajectory name": info["traj_filename"],
                        "t": [t] * len(info["traj_filename"]),
                        "Commanded linear velocity": cmd[:, 0, 0].numpy(),
                        "Commanded angular velocity": cmd[:, 0, 1].numpy(),
                        "Predicted linear velocity": out[:, 0, 0].numpy(),
                        "Predicted angular velocity": out[:, 0, 1].numpy(),
                    }
                    loss_per_traj.append(pd.DataFrame(data=data))

        loss_per_traj = pd.concat(loss_per_traj)
        loss_per_traj = loss_per_traj.sort_values(by=["Trajectory name"])

        for ax, el in zip((ax_vel, ax_ang), ("Linear", "Angular")):
            sns.scatterplot(
                data=loss_per_traj,
                y=f"Commanded {el.lower()} velocity",
                x=f"Predicted {el.lower()} velocity",
                hue="Trajectory name",
                alpha=0.4,
                size=0.05,
                ax=ax,
            )
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_xlabel("")
            if not plot_label:
                ax.set_ylabel("")
            ax.get_legend().remove()


def run_plotting(rank, cfg: DictConfig, output_dir: Path):
    data_path = Path(cfg.paths.bc_data_dir)
    # train_dataset = BCDataset(
    #     cfg.hist_size,
    #     cfg.pred_horizon,
    #     data_path,
    #     cfg.mirror_trajs,
    #     cfg.max_goal_time_sample,
    # )
    val_data_path = Path(cfg.paths.bc_val_data_dir)
    val_dataset = BCDataset(
        cfg.hist_size,
        cfg.pred_horizon,
        val_data_path,
        cfg.mirror_trajs,
        cfg.max_goal_time_sample,
    )

    mu, std = train_dataset.get_mean_std_commands()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exit(0)

    model = torch.load(
        "/media/jft/diskstation/results_bc/models_for_sept11/model_bc_dino.pth"
    )
    model.to(device)

    fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.5))

    print("===== Running validation ======")
    validation(val_dataset, model, cfg, device, mu, std, axs[0, 0], axs[0, 1], False)
    print("===== Running train validation ======")
    # val_train_dataset = train_dataset
    validation(
        train_dataset,
        model,
        cfg,
        device,
        mu,
        std,
        axs[1, 0],
        axs[1, 1],
        True,
    )
    fig.supxlabel("Predicted")
    fig.supylabel("Commanded")

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.savefig("fig.pdf")
    plt.show()


config_path = Path(__file__).parent.parent.parent / "config"


@hydra.main(
    version_base=None,
    config_path=str(config_path),
    config_name="train_bc",
)
def my_app(cfg: DictConfig) -> None:
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    return run_plotting(0, cfg, output_dir)


if __name__ == "__main__":
    my_app()
