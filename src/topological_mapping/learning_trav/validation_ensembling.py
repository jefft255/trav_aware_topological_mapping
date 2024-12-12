import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

import heapq


# from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    MulticlassPrecision,
    Recall,
    ROC,
)
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn.functional as F

mpl.rcParams["figure.dpi"] = 300


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
plt.rcParams["font.family"] = "Times New Roman"

from heapq import heapify, heappush, heappushpop, nlargest


class MaxHeap:
    def __init__(self, top_n):
        self.h = []
        self.length = top_n
        heapq.heapify(self.h)

    def add(self, var, img):
        var = var.item()
        var += 0.00001 * torch.randn((1,)).item()
        if len(self.h) < self.length:
            heapq.heappush(self.h, (var, img))
        else:
            heapq.heappushpop(self.h, (var, img))

    def add_batch(self, var, img):
        print(f"{var.shape=}")
        assert var.shape[0] == img.shape[0]
        for b in range(var.shape[0]):
            self.add(var[b], img[b])

    def get_top(self):
        return torch.stack([e[1] for e in self.h])


def model_ensemble_wrapper(models):
    # TODO make extra sure this works

    def model_fun(imgs, Ts):
        output_list = []
        for model in models:
            output_list.append(model(imgs, Ts))
        output = torch.stack(output_list)
        class_scores = F.softmax(output, dim=-1)
        avg_class_scores = torch.mean(class_scores, dim=0)
        std = torch.std(class_scores[:, :, 0], dim=0)
        return output, class_scores, avg_class_scores, std

    return model_fun


def validation_plottravmap(
    val_dataset, models, batch_size, device, logger, epoch, mode
):
    val_dataloader = DataLoader(val_dataset, batch_size, True)
    false_positives = []
    true_positives = []
    false_negatives = []
    stds = []
    heap_var = MaxHeap(10)
    models = model_ensemble_wrapper(models)
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            imgs_cpu, Ts, label, imgs_point = batch
            imgs = imgs_cpu.to(device)
            Ts = Ts.to(device)
            _, _, out, std = models(imgs, None, Ts)
            heap_var.add_batch(std, imgs)
            untrav_index = torch.argmax(out, -1) == 1
            trav_index = torch.argmax(out, -1) == 0
            label = label[:, 1].to(device)
            fpi = torch.logical_and(untrav_index, label == 0).cpu()
            tpi = torch.logical_and(untrav_index, label == 1).cpu()
            fni = torch.logical_and(trav_index, label == 1).cpu()

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
        grid = make_grid(imgs, nrow=6)
        logger.add_image(f"{mode}/{name}", grid, epoch)

    grid = make_grid(heap_var.get_top(), nrow=2)
    logger.add_image(f"{mode}/top_var", grid, epoch)


def validation_ploterrors(val_dataset, models, batch_size, device, logger, epoch, mode):
    val_dataloader = DataLoader(val_dataset, batch_size, True)
    false_positives = []
    true_positives = []
    false_negatives = []
    stds = []
    heap_var = MaxHeap(10)
    models = model_ensemble_wrapper(models)
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            imgs_cpu, Ts, label, imgs_point, pixel_onehot = batch
            imgs = imgs_cpu.to(device)
            # Ts = Ts.to(device)
            Ts = pixel_onehot.to(device)
            _, _, out, std = models(imgs, Ts)
            heap_var.add_batch(std, imgs_point)
            untrav_index = torch.argmax(out, -1) == 1
            trav_index = torch.argmax(out, -1) == 0
            label = label[:, 1].to(device)
            fpi = torch.logical_and(untrav_index, label == 0).cpu()
            tpi = torch.logical_and(untrav_index, label == 1).cpu()
            fni = torch.logical_and(trav_index, label == 1).cpu()

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
        grid = make_grid(imgs, nrow=6)
        logger.add_image(f"{mode}/{name}", grid, epoch)

    grid = make_grid(heap_var.get_top(), nrow=2)
    logger.add_image(f"{mode}/top_var", grid, epoch)


def validation(
    val_dataset, models, batch_size, device, logger, epoch, mode, output_dir
):
    val_dataloader = DataLoader(val_dataset, batch_size, True)
    outs = []
    labels = []
    models = model_ensemble_wrapper(models)
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            imgs, Ts, label, _, pixel_onehot = batch
            imgs = imgs.to(device)
            # Ts = Ts.to(device)
            Ts = pixel_onehot.to(device)
            _, _, out, _ = models(imgs, Ts)
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
    Rec = Recall(task="multiclass", average=None, num_classes=2)
    roc = ROC(task="binary", thresholds=1000)
    accuracy = Acc(out.cpu(), labels_01.cpu())
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
