import torch
from topological_mapping.learning_trav.dataset import TraversabilityDataset
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from topological_mapping.learning_trav.models import (
    DINOv2TraversabilityAnalyser,
    ResNetTraversabilityAnalyser,
)
from topological_mapping.learning_trav.validation_ensembling import (
    validation,
    validation_ploterrors,
)

import hydra
from omegaconf import DictConfig, OmegaConf
import socket
import os
import shutil


def prepare_narval_data(orig_data_path: Path):
    data_path_root = Path(os.environ.get("SLURM_TMPDIR")) / "trav_data"
    try:
        os.mkdir(str(data_path_root))
    except FileExistsError:
        pass
    for archive in orig_data_path.glob("*.tar.gz"):
        print(f"Copying {archive}...")
        shutil.copyfile(str(archive), data_path_root / archive.name)
        print(f"Decompressing {archive}...")
        os.system(f"tar -zxf {data_path_root / archive.name} -C {data_path_root}")
    return data_path_root


if socket.gethostname() == "raza":
    config_path = "/home/adaptation/jft/catkin_ws/src/topological_mapping/config"
else:
    config_path = "../../../../topological_mapping/config"


@hydra.main(
    version_base=None,
    config_path=config_path,
    config_name="train_trav_ensembling",
)
def my_app(cfg: DictConfig) -> None:
    # torch.backends.cuda.matmul.allow_tf32 = (
    #     True  # Enable/disable TF32 for matrix multiplications
    # )
    # torch.backends.cudnn.allow_tf32 = True  # Enable/disable TF32 for convolutions
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logger = SummaryWriter(output_dir)
    if "narval" in socket.gethostname():
        data_path_root = prepare_narval_data(Path(cfg.data_path))
    else:
        data_path_root = Path(cfg.data_path)

    manual_labeling_path = data_path_root.glob("**/labeling_*_labeled")
    manual_labeling_path = [p for p in manual_labeling_path if p.is_dir()]

    # manual_labeling_path = []
    dataset = TraversabilityDataset(
        manual_labeling_path,
        [],
    )

    train_dataset, val_dataset = random_split(
        dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    )
    if cfg.tiny_dataset:
        train_dataset, _ = random_split(train_dataset, [0.2, 0.8])

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
    models = []
    for model_i in range(cfg.n_models):
        print(f"================ Training model {model_i} ====================")
        train_dataset, _ = random_split(dataset, [0.6, 0.4])
        if cfg.dino:
            model = DINOv2TraversabilityAnalyser(
                cfg.hidden_dim, cfg.n_layers, cfg.version, cfg.one_hot
            )
        else:
            model = ResNetTraversabilityAnalyser(
                cfg.hidden_dim, cfg.n_layers, cfg.version, cfg.one_hot
            )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)
        for epoch in range(cfg.n_epochs):
            print(f"===== Epoch {epoch} =====")
            avg_loss = 0.0
            for batch in tqdm(train_dataloader):
                imgs, Ts, labels, _, pixels = batch
                if cfg.one_hot:
                    Ts = pixels
                imgs = imgs.to(device)
                Ts = Ts.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                out = model(imgs, Ts)
                loss = F.binary_cross_entropy_with_logits(out, labels)
                avg_loss += loss.item()
                loss.backward()
                optimizer.step()
            logger.add_scalar(
                f"Loss/train_{model_i}", avg_loss / len(train_dataloader), epoch
            )

            if epoch % cfg.eval_every_epoch == 0:
                torch.save(model, open(output_dir / f"model_{model_i}.pth", "wb"))
        models.append(model)
    validation(
        val_dataset, models, cfg.batch_size, device, logger, 0, "validation", output_dir
    )
    validation_ploterrors(
        val_dataset, models, cfg.batch_size, device, logger, 0, "validation"
    )


if __name__ == "__main__":
    my_app()
