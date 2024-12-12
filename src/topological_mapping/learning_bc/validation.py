import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from math import sin, cos

sns.set_theme()
# plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["figure.facecolor"] = "#f7f7f2"
plt.rcParams["axes.facecolor"] = "#f7f7f2"
plt.rcParams["axes.edgecolor"] = "#cecdca"
plt.rcParams["grid.color"] = "#cecdca"


def plot_dataset(train_dataloader, logger):
    datas = []
    for batch in tqdm(train_dataloader):
        imgs_l, imgs_f, imgs_r, data_vels, past_cmds, goal, cmd, info = batch
        data = {
            "Trajectory name": info["traj_filename"],
            "Commanded linear velocity": cmd[:, 0, 0].numpy(),
            "Commanded angular velocity": cmd[:, 0, 1].numpy(),
            "Past commanded lin. vel.": past_cmds[:, 0, 0].numpy(),
            "x velocity": data_vels[:, 2, 0, 3].numpy() * 5,
            "y velocity": data_vels[:, 2, 1, 3].numpy() * 5,
            "z velocity": data_vels[:, 2, 2, 3].numpy() * 5,
            "goal x": goal[:, 0],
            "goal y": goal[:, 1],
        }
        datas.append(pd.DataFrame(data=data))
    data = pd.concat(datas, ignore_index=True)
    nan_rows = data[data.isnull().T.any()]
    if len(nan_rows) > 0:
        raise ValueError("The data has NaNs! Fix the data!")

    data = data.sort_values(by=["Trajectory name"])

    fig = plt.figure(figsize=(8.0, 8.0), dpi=150)
    ax = fig.gca()
    sns.scatterplot(
        data=data,
        y="x velocity",
        x="Past commanded lin. vel.",
        hue="Trajectory name",
        alpha=0.6,
        ax=ax,
    )
    ax.get_legend().remove()
    fig.tight_layout()
    logger.add_figure(f"dataset_vel_delta", fig, 0)

    fig = plt.figure(figsize=(8.0, 8.0), dpi=150)
    ax = fig.gca()
    sns.scatterplot(
        data=data,
        y="goal y",
        x="goal x",
        hue="Trajectory name",
        alpha=0.6,
        ax=ax,
    )
    ax.get_legend().remove()
    fig.tight_layout()
    logger.add_figure(f"dataset_gals", fig, 0)

    for dim in ["x", "y", "z"]:
        fig = plt.figure(figsize=(24.0, 8.0), dpi=150)
        ax = fig.gca()
        sns.violinplot(
            data=data,
            y=f"{dim} velocity",
            x="Trajectory name",
            ax=ax,
            orient="v",
            inner=None,
            cut=0,
            linewidth=0.1,
        )
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        logger.add_figure(f"dataset_vel_{dim}", fig, 0)

    for dim in ["linear", "angular"]:
        fig = plt.figure(figsize=(12.0, 8.0), dpi=150)
        ax = fig.gca()
        sns.violinplot(
            data=data,
            y=f"Commanded {dim} velocity",
            x="Trajectory name",
            ax=ax,
            orient="v",
            inner=None,
            cut=0,
            linewidth=0.1,
        )
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        logger.add_figure(f"dataset_cmd_{dim}", fig, 0)


def validation(val_dataset, model, logger, epoch, cfg, device, mu, std, mode):
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
            # IMPORTANT HERE; different loss, than optimized loss!
            # We're only looking at the first timestep prediction error
            # loss here because that's what we're actually executing
            loss = F.mse_loss(out[:, 0], cmd[:, 0])
            avg_loss += loss.item()

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
                    data["Linear velocity loss"] = torch.pow((out - cmd), 2)[
                        :, t, 0
                    ].numpy()
                    data["Angular velocity loss"] = torch.pow((out - cmd), 2)[
                        :, t, 1
                    ].numpy()
                    loss_per_traj.append(pd.DataFrame(data=data))

        loss_per_traj = pd.concat(loss_per_traj)
        loss_per_traj = loss_per_traj.sort_values(by=["Trajectory name"])

        for el in ("Linear", "Angular"):
            fig = plt.figure(figsize=(12.0, 8.0), dpi=150)
            ax = fig.gca()
            sns.violinplot(
                data=loss_per_traj,
                y=f"{el} velocity loss",
                x="Trajectory name",
                ax=ax,
                orient="v",
                inner=None,
                hue="t",
                cut=0,
                linewidth=0.1,
            )
            ax.tick_params(axis="x", rotation=90)
            fig.tight_layout()
            logger.add_figure(f"loss_{mode}_{el}_per_traj", fig, epoch)

            fig = plt.figure(figsize=(3.5, 3.5), dpi=300)
            ax = fig.gca()
            sns.scatterplot(
                data=loss_per_traj,
                y=f"Commanded {el.lower()} velocity",
                x=f"Predicted {el.lower()} velocity",
                hue="Trajectory name",
                alpha=0.4,
                size=0.05,
                ax=ax,
            )
            ax.get_legend().remove()
            fig.tight_layout()
            logger.add_figure(f"reg_{mode}_{el}", fig, epoch)
        print(avg_loss / len(val_dataloader))
        logger.add_scalar(f"Loss/val_{mode}", avg_loss / len(val_dataloader), epoch)
        model.train()
        return avg_loss / len(val_dataloader)


def validation_goal(val_dataset, model, logger, epoch, cfg, device, mu, std):
    with torch.no_grad():
        model.eval()
        val_dataloader = DataLoader(val_dataset, 64, True)
        i = 0
        n_thingy = 3
        fig, axs = plt.subplots(n_thingy * 4, 4, dpi=300)
        fig.set_size_inches(4, 12)
        for batch in tqdm(val_dataloader):
            imgs_l, imgs_f, imgs_r, vels, past_cmds, goal, cmd, info = batch
            past_cmds = (past_cmds - mu) / std
            past_cmds = past_cmds[:4].to(device)
            imgs_l = imgs_l[:4].to(device)
            imgs_f = imgs_f[:4].to(device)
            imgs_r = imgs_r[:4].to(device)
            vels = vels[:4].to(device)
            goal = goal[:4]
            goal = goal.to(device)
            out = model(imgs_l, imgs_f, imgs_r, vels, past_cmds, goal)
            out = std * out.cpu() + mu
            for j in range(4):
                axs[4 * i + j, 3].arrow(
                    0,
                    0,
                    out[j, 0, 0] * sin(out[j, 0, 1]),
                    out[j, 0, 0] * cos(out[j, 0, 1]),
                    color="yellow",
                    label="output",
                )
                axs[4 * i + j, 3].arrow(
                    0, 0, goal[j, 1].cpu(), goal[j, 0].cpu(), color="red", label="goal"
                )
                axs[4 * i + j, 3].arrow(
                    0,
                    0,
                    cmd[j, 0, 1].cpu(),
                    cmd[j, 0, 0].cpu(),
                    color="blue",
                    label="gt",
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
        model.train()
