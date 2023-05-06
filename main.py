import argparse
import gc
import logging
import tempfile
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import utils
from model import UNet_CutPaste_AD

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)


# CHECK:utilsへ移行？
def parse_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "--exp_name",
        type=str,
        default=r"test",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=r"debug",
    )

    parser.add_argument(
        "--mlflow_path",
        type=str,
        default=r"",
    )

    parser.add_argument(
        "--result_dir",
        type=Path,
        default=r"result",
    )

    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=r"",
    )

    parser.add_argument("--seed", default=1024)
    parser.add_argument("--nb_epochs", type=int, default=1)
    parser.add_argument("--input_size", default=100)
    args = parser.parse_args()

    args.exp_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    return args


def main():

    cfg = parse_args()
    logger = utils.get_logger()

    mlflow.set_tracking_uri(cfg.mlflow_path)
    mlflow.set_experiment(cfg.exp_name)
    tracking = mlflow.tracking.MlflowClient()
    logger.debug(f"exp_name:{cfg.exp_name}, data_path:{cfg.dataset_dir}")
    print(cfg.exp_name)
    experiment = tracking.get_experiment_by_name(cfg.exp_name)
    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=cfg.run_name, nested=True
    ):

        utils.set_seeds(use_cuda, cfg.seed)
        # モデルの設定の読み込み
        model = UNet_CutPaste_AD()
        model.to(device)
        train_dataloader, test_dataloader = utils.get_cutpaste_dataloader(
            cfg.dataset_dir, input_size=(224, 224), batch_size=32
        )
        # 学習
        model.fit(train_dataloader, nb_epochs=cfg.nb_epochs)
        model.save(cfg.result_dir / "model.pht")

        test_imgs, img_paths, gts, gt_masks, scores = model.evaluate(test_dataloader)

        # NOTE: 正規化について検討
        # Normalization
        # max_score = score_map.max()
        # min_score = score_map.min()
        # scores = (score_map - min_score) / (max_score - min_score)
        # mlflow.log_metric("max_score", max_score)
        # mlflow.log_metric("min_score", min_score)

        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gts = np.asarray(gts)
        gt_masks = np.asarray(gt_masks)

        # 閾値の設定
        threshold = utils.get_optimal_threshold(img_scores, gts)
        mlflow.log_metric("threshold", threshold)

        # 画像レベルので評価
        fpr, tpr, thresholds = roc_curve(gts, img_scores, drop_intermediate=False)
        min_fpr = min(fpr[np.where(tpr == 1.0)])
        logger.debug(f"min fpr@tpr=1:{min_fpr}")
        mlflow.log_metric("min fpr_tpr_1", min_fpr)

        img_roc_auc = roc_auc_score(gts, img_scores)
        logger.debug(f"image_roc_auc:{img_roc_auc}")
        mlflow.log_metric("image_roc_auc", img_roc_auc)

        utils.plot_roc_curve(
            fpr,
            tpr,
            img_roc_auc,
            cfg.result_dir.joinpath("img_roc_curve.png"),
            "img_ROCAUC",
        )

        cm_path = cfg.result_dir.joinpath("cm.png")
        y_pred = [0 if score < threshold else 1 for score in img_scores]
        recall, precision, accuracy, f1, cm = utils.calc_metrics(gts.flatten(), y_pred)
        cm = confusion_matrix(gts, y_pred)
        logger.debug(cm)
        logger.debug(classification_report(gts, y_pred))

        utils.plot_cm(cm, cm_path)
        mlflow.log_metric("image_roc_auc", img_roc_auc)

        fpr, tpr, thresholds = roc_curve(gt_masks.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_masks.flatten(), scores.flatten())
        logger.debug(f"per_pixel_rocauc:{per_pixel_rocauc}")
        utils.plot_roc_curve(
            fpr,
            tpr,
            per_pixel_rocauc,
            cfg.result_dir.joinpath("pixel_roc_curve.png"),
            "pixel_ROCAUC",
        )

        pixel_cm_path = cfg.result_dir.joinpath("pixel_cm.png")
        y_pred = [0 if score < threshold else 1 for score in scores.flatten()]
        recall, precision, accuracy, f1, cm = utils.calc_metrics(
            gt_masks.flatten(), y_pred
        )
        utils.plot_cm(
            cm,
            pixel_cm_path,
        )

        # CHECK: コメントを英語に
        # 異常度マップなどのプロット
        utils.plot_results(
            test_imgs,
            scores,
            gt_masks,
            threshold,
            cfg.result_dir,
            img_paths,
        )


if __name__ == "__main__":
    main()
