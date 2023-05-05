from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

import utils
from unet import UNet

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# CHECK: 名前の変更
class UNet_CutPaste_AD:
    def __init__(self):
        # TODO: 引数にする
        self.model = UNet(3, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = torch.nn.BCELoss()

    def fit(self, train_loader, nb_epochs=1):
        self.model.train()
        train_loss = utils.AverageMeter("train_loss")
        for epoch in range(nb_epochs):
            for (img, gt) in train_loader:
                self.optimizer.zero_grad()
                img = img.to(device)
                output = self.model(img)
                loss = self.criterion(output, gt)
                loss.backward()
                self.optimizer.step()
                train_loss.update(loss.item(), img.shape[0])
                print("epoch:", epoch, train_loss)

    def evaluate(self, test_loader):
        gts, gt_masks, test_imgs, img_paths, score_maps = [], [], [], [], []
        self.model.eval()
        for (x, y, mask, img_path) in tqdm(test_loader, "test"):
            test_imgs.extend(x.cpu().detach().numpy())
            gts.extend(y.cpu().detach().numpy())
            gt_masks.extend(mask.cpu().detach().numpy())
            img_paths.extend(list(img_path))

            score_map = torch.squeeze(self.model(x), dim=1)
            score_maps.extend(score_map.detach().numpy())
        score_maps = np.array(score_maps)

        for i in range(score_maps.shape[0]):
            score_maps[i] = gaussian_filter(score_maps[i], sigma=4)
        return test_imgs, img_paths, gts, gt_masks, score_maps

    def to(self, device):
        self.model.to(device)

    def save(self, result_path: Path):
        torch.save(self.model.state_dict(), result_path)
