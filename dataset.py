import os
import cv2
import torch
from torch.utils.data import Dataset


class ClothesDataset(Dataset):
    def __init__(self, cloth_dir, mask_dir):
        self.cloth_dir = cloth_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(cloth_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        cloth_path = os.path.join(self.cloth_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        cloth = cv2.imread(cloth_path)
        cloth = cv2.cvtColor(cloth, cv2.COLOR_BGR2RGB)
        cloth = cv2.resize(cloth, (256, 256))

        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (256, 256))

        cloth = torch.tensor(cloth).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask).unsqueeze(0) / 255.0

        return cloth.float(), mask.float()
