import gc
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from skimage import morphology
from skimage.segmentation import mark_boundaries
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader
from typing import List

from dataset import CutPasteDataset

plt.rc("font", size=15)
logger = logging.getLogger("logger")


def get_optimal_threshold(scores: np.ndarray, gt_mask: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(
        gt_mask.flatten(), scores.flatten()
    )
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    return threshold


def get_cutpaste_dataloader(
    data_path,
    input_size=(224, 224),
    batch_size=32,
):
    # 良品学習用のデータローダーの取得
    train_dataset = CutPasteDataset(
        data_path,
        mode="train",
        image_size=input_size,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True
    )

    test_dataset = CutPasteDataset(
        data_path,
        mode="test",
        image_size=input_size,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False
    )
    return train_dataloader, test_dataloader


def get_logger():
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s"
    )
    stream_handler.setFormatter(fh_formatter)
    logger.addHandler(stream_handler)
    return logger


def denormalization(x:np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.0).astype(np.uint8)
    return x


def set_seeds(use_cuda:bool, seed:int=1024):
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def calc_metrics(y_test: np.ndarray, y_pred: np.ndarray):
    recall = recall_score(y_test, y_pred)
    logger.debug(f"Recall : {recall:.3f}")
    precision = precision_score(y_test, y_pred)
    logger.debug(f"Precision : {precision:.3f}")
    accuracy = accuracy_score(y_test, y_pred)
    logger.debug(f"Accuracy : {accuracy:.3f}")
    f1 = f1_score(y_test, y_pred)
    logger.debug(f"F1 Score: {f1:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    # logger.debug(cm)
    # logger.debug(classification_report(y_test, y_pred))

    return recall, precision, accuracy, f1, cm


def plot_cm(cm:np.ndarray, fig_path:Path):

    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
    disp.plot(cmap="Greens")
    plt.savefig(fig_path)
    plt.close()
    mlflow.log_artifact(fig_path)


def plot_roc_curve(fpr:np.ndarray, tpr:np.ndarray, _auc:float, fig_path:str="roc.png", title:str="rocauc"):
    plt.figure()
    plt.title(title)
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % _auc)
    plt.plot([0, 1], [0, 1], "r--", label="random")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig(fig_path)
    plt.close()
    mlflow.log_artifact(fig_path)


def plot_results(
    imgs:List,
    scores:np.ndarray,
    gts:np.ndarray,
    threshold:float,
    result_dir:Path,
    img_paths:List,
):

    num_imgs = len(scores)
    vmax = 255.0
    vmin = 0.0

    for i in range(num_imgs):
        img = imgs[i]
        img = denormalization(img)
        if gts[i].ndim != 2:
            gt = gts[i].transpose(1, 2, 0).squeeze()
        else:
            gt = gts[i]
        img_score = np.max(scores[i])
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask >= threshold] = 1
        mask[mask < threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode="thick")
        fig_img, ax_img = plt.subplots(1, 3, figsize=(40, 20))
        # fig_img, ax_img = plt.subplots(3, 1, figsize=(60, 30))
        fig_img.subplots_adjust(right=0.9)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        img_gt = mark_boundaries(img, gt, color=(1, 0, 0), mode="thick")
        ax_img[0].imshow(img)
        ax_img[0].title.set_text("Image")
        ax_img[1].imshow(img_gt)
        ax_img[1].title.set_text("Ground truth")
        ax_img[2].imshow(img, cmap="gray", interpolation="none")
        ax = ax_img[2].imshow(
            heat_map, cmap="jet", alpha=0.5, interpolation="none", vmin=vmin, vmax=vmax
        )
        ax_img[2].title.set_text("Predicted heat map")
        # ax_img[3].imshow(vis_img)
        # ax_img[3].title.set_text("Segmentation result")
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            "family": "serif",
            "color": "black",
            "weight": "normal",
            "size": 8,
        }
        cb.set_label("Anomaly Score", fontdict=font)

        img_path = Path(img_paths[i])
        img_basename = img_path.stem
        category = img_path.parts[-2]

        fig_path = result_dir.joinpath(
            "{}_{}_{}.png".format(category, img_basename, str(round(img_score, 3)))
        )
        fig_img.savefig(fig_path, dpi=100)
        plt.close()
        mlflow.log_artifact(fig_path)

        gc.collect()


class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
