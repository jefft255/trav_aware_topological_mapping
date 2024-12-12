from pathlib import Path
from tqdm import tqdm
import sys
import traceback

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from topological_mapping.learning_bc.dataset import BCDataset
from topological_mapping.learning_bc.models import DINOv2BCNet, ResNetBCNet
from topological_mapping.learning_bc.validation import (
    validation,
    validation_goal,
    plot_dataset,
)
from topological_mapping.learning_bc.validation_sim import validation_simulator

import hydra
from omegaconf import DictConfig
import os
import time


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def precompute_embeddings(dataset: BCDataset, encoder, device):
    with torch.no_grad():
        print("Precomputing embeddings...")
        for traj in dataset.trajs:
            embs = []
            # Left front right order is important
            for side in ["l", "f", "r"]:
                print(f"Doing {side} side")
                imgs = traj[f"images_{side}"]
                i = 0
                step = 64
                embeddings = []
                while i < imgs.shape[0]:
                    end = min(i + step, imgs.shape[0])
                    img_batch = dataset.process_img_list(imgs[i:end]).to(device)
                    embeddings.append(encoder(img_batch).cpu())
                    i += step
                embs_for_side = torch.concatenate(embeddings, dim=0)
                print(f"{embs_for_side.shape=}")
                embs.append(embs_for_side)
            traj[f"embs"] = torch.concatenate(embs, dim=-1)


def train(rank, cfg: DictConfig, output_dir: Path, logger: SummaryWriter):
    if cfg.distributed:
        setup(rank, cfg.world_size)

    torch.multiprocessing.set_sharing_strategy("file_system")

    data_path = Path(cfg.paths.bc_data_dir)
    train_dataset = BCDataset(
        cfg.hist_size,
        cfg.pred_horizon,
        data_path,
        cfg.mirror_trajs,
        cfg.max_goal_time_sample,
    )
    val_data_path = Path(cfg.paths.bc_val_data_dir)
    val_dataset = BCDataset(
        cfg.hist_size,
        cfg.pred_horizon,
        val_data_path,
        cfg.mirror_trajs,
        cfg.max_goal_time_sample,
    )

    mu, std = train_dataset.get_mean_std_commands()
    if cfg.dataset_proportion < 1.0:
        train_dataset, _ = random_split(
            train_dataset, [cfg.dataset_proportion, 1.0 - cfg.dataset_proportion]
        )

    if cfg.distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=cfg.world_size,
            rank=rank,
            drop_last=True,
            shuffle=True,
        )
    else:
        sampler = None
    train_dataloader = DataLoader(
        train_dataset,
        cfg.batch_size,
        shuffle=sampler is None,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        sampler=sampler,
        drop_last=True,
        # pin_memory=True,
    )
    plot_dataset(train_dataloader, logger)

    device = rank

    if cfg.DINO:
        modelcls = DINOv2BCNet
    else:
        modelcls = ResNetBCNet
    model = modelcls(
        mu,
        std,
        version=cfg.version,
        hist_size=cfg.hist_size,
        pred_horizon=cfg.pred_horizon,
        only_front=cfg.only_front,
        past_actions=cfg.past_actions,
        velocities=cfg.velocities,
        hidden_layers=cfg.hidden_layers,
        hidden_dim=cfg.hidden_dim,
        gelu=cfg.gelu,
        residual=cfg.residual,
    )
    model.to(device)

    if cfg.precompute_embeddings:
        assert cfg.DINO
        precompute_embeddings(train_dataset, model.img_enc, device)
        train_dataset.get_images = False
    if cfg.distributed:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)

    time_begin_epoch = time.time()

    global_step = 0
    for epoch in range(cfg.n_epochs):
        print(f"===== Epoch {epoch} =====")
        if cfg.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        avg_loss = 0.0
        for batch in tqdm(train_dataloader):
            imgs_l, imgs_f, imgs_r, vels, past_cmds, goal, cmd, info = batch
            past_cmds = (past_cmds - mu) / std
            cmd = (cmd - mu) / std
            if cfg.precompute_embeddings:
                embs = info["embs"].to(device)
            else:
                imgs_l = imgs_l.to(device)
                imgs_f = imgs_f.to(device)
                imgs_r = imgs_r.to(device)
                embs = None
            vels = vels.to(device)
            goal = goal.to(device)
            past_cmds = past_cmds.to(device)
            cmd = cmd.to(device)

            optimizer.zero_grad()
            out = model(imgs_l, imgs_f, imgs_r, vels, past_cmds, goal, embs)
            assert out.shape == cmd.shape
            loss = F.mse_loss(out, cmd)

            avg_loss += loss.item()
            logger.add_scalar(
                "Loss/train_per_batch", loss.item(), global_step=global_step
            )
            global_step += 1
            loss.backward()
            # print(model.fuser.first_layer.weight.grad)
            optimizer.step()

        time_end_epoch = time.time()
        if time_end_epoch - time_begin_epoch > cfg.timeout_epoch:
            raise TimeoutError(
                "Training epoch is taking too long. Ending run for sweep."
            )
        time_begin_epoch = time.time()

        if rank == 0:
            logger.add_scalar("Loss/train", avg_loss / len(train_dataloader), epoch)
            print(avg_loss / len(train_dataloader))

            if epoch % 10 == 0:
                train_dataset.get_images = True
                print("===== Running goal validation ======")
                last_val_error = validation_goal(
                    val_dataset, model, logger, epoch, cfg, device, mu, std
                )
                print("===== Running validation ======")
                val_val_dataset, _ = random_split(val_dataset, [0.3, 0.7])
                validation(
                    val_val_dataset, model, logger, epoch, cfg, device, mu, std, "val"
                )
                print("===== Running train validation ======")
                val_train_dataset, _ = random_split(train_dataset, [0.03, 0.97])
                # val_train_dataset = train_dataset
                validation(
                    val_train_dataset,
                    model,
                    logger,
                    epoch,
                    cfg,
                    device,
                    mu,
                    std,
                    "train",
                )
                if cfg.distributed:
                    model_to_save = model.module
                else:
                    model_to_save = model
                filename = (
                    output_dir / f"model_bc_{'dino' if cfg.DINO else 'resnet'}.pth"
                )
                torch.save(model_to_save, open(filename, "wb"))
                if cfg.precompute_embeddings:
                    train_dataset.get_images = False

    if rank == 0 and cfg.run_sim:
        # Make sure to free up resources for the simulator
        del model
        del model_to_save
        del train_dataloader
        del train_dataset
        # Return the mean success rate for Optuna sweeps
        return validation_simulator(output_dir, cfg, logger, epoch)

    if cfg.distributed:
        cleanup()

    return last_val_error


config_path = Path(__file__).parent.parent.parent.parent / "config"


@hydra.main(
    version_base=None,
    config_path=str(config_path),
    config_name="train_bc",
)
def my_app(cfg: DictConfig) -> None:
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger = SummaryWriter(output_dir)
    logger.add_text("Hyperparams", str(cfg), 1)

    if cfg.distributed:
        mp.spawn(
            train,
            args=(
                cfg,
                output_dir,
                logger,
            ),
            nprocs=cfg.world_size,
            join=True,
        )
    else:
        try:
            return train(0, cfg, output_dir, logger)
        except Exception as e:
            error_txt = "".join(traceback.TracebackException.from_exception(e).format())
            logger.add_text("Error", error_txt, 1)
            # logger.add_scalar("successrate_mean", -1, cfg.n_epochs - 1)
            # logger.add_scalar("successrate_std", 0, cfg.n_epochs - 1)
            print(f"Error: {error_txt}", file=sys.stderr)
            # Return -1 for hyperparam opt
            return -1


if __name__ == "__main__":
    my_app()
