import torch
from topological_mapping.learning_trav.dataset import TraversabilityDataset
from pathlib import Path
from tqdm import tqdm
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import random_split
from topological_mapping.learning_trav.models import (
    DINOv2TraversabilityAnalyser,
    ResNetTraversabilityAnalyser,
)
from topological_mapping.utils import project_node
import torch.nn.functional as F

# from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    MulticlassPrecision,
    Recall,
    ROC,
    MulticlassConfusionMatrix,
)
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import hydra
from omegaconf import DictConfig, OmegaConf
import socket
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib as mpl

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


def validation_plottravmap(val_dataset, model, device, logger, epoch, mode, cfg):
    val_dataloader = DataLoader(val_dataset, cfg.batch_size, True)
    n_points = 224
    depth_dists = [1.0, 2.0, 3.0]
    n_images = 25
    grid = []
    i = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            imgs_cpu, _, _, _, _ = batch
            for b in range(imgs_cpu.shape[0]):
                if i >= n_images:
                    break
                img_cpu = imgs_cpu[b]
                outs = []
                for depth_dist in depth_dists:
                    if cfg.one_hot:
                        n_points = (224 // 14) ** 2
                        Ts = depth_dist * torch.eye(n_points)
                    else:
                        Ts_x = torch.linspace(-2, 2, n_points)
                        Ts_y = torch.linspace(0, 1.5, n_points)
                        Ts = torch.cartesian_prod(Ts_y, Ts_x)
                        Ts = torch.concatenate(
                            [Ts, depth_dist * torch.ones(n_points**2, 1)], dim=-1
                        )
                    img = img_cpu.to(device)
                    Ts = Ts.to(device)
                    out = model.multi_goal_forward(img, Ts)
                    out = F.softmax(out, dim=-1).cpu()
                    if cfg.one_hot:
                        out = out[:, 0].reshape(1, 224 // 14, 224 // 14)
                        out = torchvision.transforms.functional.resize(out, (224, 224))
                    else:
                        out = out[:, 1].reshape(n_points, n_points)
                    outs.append(out.cpu())
                out_img = torch.concatenate(outs, dim=0)
                # img_cpu = val_dataset.untransform(img_cpu)
                out_img = torch.concatenate([out_img, img_cpu], dim=-1)
                grid.append(out_img)
                i += 1
    grid = make_grid(torch.stack(grid), nrow=5)

    logger.add_image(f"{mode}/trav_map", grid, epoch)


def validation_ploterrors(
    val_dataset,
    model,
    device,
    logger,
    epoch,
    mode,
    cfg,
):
    val_dataloader = DataLoader(val_dataset, cfg.batch_size, True)
    false_positives = []
    true_positives = []
    false_negatives = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            imgs_cpu, Ts, label, imgs_point, pixels = batch
            if cfg.one_hot:
                Ts = pixels
            imgs = imgs_cpu.to(device)
            Ts = Ts.to(device)
            out = model(imgs, Ts)
            out = F.softmax(out, dim=-1).cpu()
            untrav_index = torch.argmax(out, -1) == 1
            trav_index = torch.argmax(out, -1) == 0
            label = label[:, 1]
            fpi = torch.logical_and(untrav_index, label == 0)
            tpi = torch.logical_and(untrav_index, label == 1)
            fni = torch.logical_and(trav_index, label == 1)

            false_positives.append(imgs_point[fpi])
            true_positives.append(imgs_point[tpi])
            false_negatives.append(imgs_point[fni])

    for name, imgs in zip(
        ["False untrav", "True untrav", "False trav"],
        [false_positives, true_positives, false_negatives],
    ):
        imgs = torch.cat(imgs, dim=0)
        if imgs.shape[0] == 0:
            print(f"No {name} found")
            continue
        imgs = imgs[:36]
        # for i in range(imgs.shape[0]):
        #     imgs[i] = val_dataset.untransform(imgs[i])
        grid = make_grid(imgs, nrow=6)
        logger.add_image(f"{mode}/{name}", grid, epoch)


def validation(val_dataset, model, device, logger, epoch, mode, output_dir, cfg):
    val_dataloader = DataLoader(val_dataset, cfg.batch_size, True)
    outs = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            imgs, Ts, label, _, pixels = batch
            if cfg.one_hot:
                Ts = pixels
            imgs = imgs.to(device)
            Ts = Ts.to(device)
            out = model(imgs, Ts)
            out = F.softmax(out, dim=-1)
            outs.append(out)
            labels.append(label.cpu())
    out = torch.cat(outs, dim=0)
    labels = torch.cat(labels, dim=0)
    labels_01 = torch.zeros(labels.shape[0], dtype=torch.long)
    labels_01[labels[:, 1] == 1.0] = 1
    # bcm = MulticlassConfusionMatrix(2, validate_args=True)
    # logger.add_scalar("Loss/train", avg_loss / len(train_dataloader), epoch)
    Acc = Accuracy(task="multiclass", num_classes=2, average=None)
    F1 = F1Score(task="multiclass", num_classes=2, average=None)
    Mcp = MulticlassPrecision(num_classes=2, average=None)
    conf = MulticlassConfusionMatrix(num_classes=2)
    Rec = Recall(task="multiclass", average=None, num_classes=2)
    roc = ROC(task="binary", thresholds=1000)
    accuracy = Acc(out.cpu(), labels_01.cpu())
    conf = conf(out.cpu(), labels_01.cpu())
    f1 = F1(out.cpu(), labels_01.cpu())
    mcp = Mcp(out.cpu(), labels_01.cpu())
    recall = Rec(out.cpu(), labels_01.cpu())
    out_roc = out[:, 1].cpu()
    _ = roc.update(out_roc, labels_01.cpu())
    fig_roc, ax = roc.plot(score=True)
    ax.set_title(f"Traversability ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    fig_roc.set_size_inches(3.5, 3.5)
    fig_roc.tight_layout()
    fig_roc.savefig(output_dir / f"roc_{mode}_{epoch}.pdf")
    for i in range(2):
        logger.add_scalar(f"Accuracy {i}/{mode}", accuracy[i], epoch)
        logger.add_scalar(f"F1 Score {i}/{mode}", f1[i], epoch)
        logger.add_scalar(f"Percision {i}/{mode}", mcp[i], epoch)
        logger.add_scalar(f"Recall {i}/{mode}", recall[i], epoch)
    logger.add_figure(f"ROC/{mode}", fig_roc, epoch)
    torch.save(conf, output_dir / f"conf_{mode}_{epoch}.pth")


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
    config_name="train_trav",
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
    test_data_path_root = Path(cfg.test_data_path)

    manual_labeling_path = data_path_root.glob("**/labeling_*_labeled")
    manual_labeling_path = [p for p in manual_labeling_path if p.is_dir()]

    manual_labeling_path_phase4 = [
        data_path_root / "labeling_june27_second_exploration_attempt_rock_labeled",
        data_path_root / "labeling_june27_aeos_lower_1_labeled",
    ]

    # test_labeling_path = test_data_path_root.glob("**/labeling_*_labeled")
    test_labeling_pathA = [test_data_path_root / "labeling_testA_labeled"]
    test_labeling_pathA = [p for p in test_labeling_pathA if p.is_dir()]

    test_labeling_pathB = [test_data_path_root / "labeling_testB_labeled"]
    test_labeling_pathB = [p for p in test_labeling_pathB if p.is_dir()]

    # ss_labeling_path = Path("/media/jft/diskstation/bagfiles/gault_jul23/").glob("2024*")
    ss_labeling_path = []

    train_dataset = TraversabilityDataset(
        manual_labeling_path + test_labeling_pathB,
        [],
        train=True,
    )
    train_dataset_for_val = TraversabilityDataset(
        manual_labeling_path_phase4,
        [],
        train=False,
    )
    print(len(train_dataset_for_val))
    exit(0)
    test_datasetA = TraversabilityDataset(
        test_labeling_pathA,
        [],
        train=False,
    )
    test_datasetB = TraversabilityDataset(
        test_labeling_pathB,
        [],
        train=False,
    )

    # train_dataset, val_dataset = random_split(
    #     train_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    # )
    if cfg.balanced_sampling:
        sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset))
    else:
        sampler = None

    # if cfg.tiny_dataset:
    #     train_dataset, _ = random_split(train_dataset, [0.2, 0.8])

    train_dataloader = DataLoader(
        train_dataset,
        cfg.batch_size,
        sampler is None,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        sampler=sampler,
        # pin_memory=True,
    )
    device = torch.device("cuda")
    if cfg.dino:
        model = DINOv2TraversabilityAnalyser(
            cfg.hidden_dim, cfg.n_layers, cfg.version, cfg.one_hot
        )
    else:
        model = ResNetTraversabilityAnalyser(
            cfg.hidden_dim, cfg.n_layers, cfg.version, cfg.one_hot
        )

    if cfg.pretrained_dimred:
        bc_model = torch.load(cfg.pretrained_dimred_path)
        model.dim_reduction = bc_model.img_enc.dim_reduction
        model.pretrained_dimred = True

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)
    print(f"{model.modules=}")
    print(f"{model.named_modules=}")
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
        logger.add_scalar("Loss/train", avg_loss / len(train_dataloader), epoch)

        if epoch % cfg.eval_every_epoch == 0:
            for name, dataset in zip(
                # ["train", "validation", "test"],
                # [train_dataset, val_dataset, test_dataset],
                ["train", "testA", "testB"],
                [train_dataset_for_val, test_datasetA, test_datasetB],
            ):
                print(f"===== Running {name} eval ======")
                validation(
                    dataset,
                    model,
                    device,
                    logger,
                    epoch,
                    name,
                    output_dir,
                    cfg,
                )
                # validation_ploterrors(dataset, model, device, logger, epoch, name, cfg)
                validation_plottravmap(dataset, model, device, logger, epoch, name, cfg)
            torch.save(model, open(output_dir / "model.pth", "wb"))


if __name__ == "__main__":
    my_app()
