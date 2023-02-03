from pathlib import Path

import cv2
import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        super(Dataset, self).__init__()
        self.data_dir = Path(data_dir)
        self.path_to_imgs = self.data_dir.glob("**/*.png")

    def __getitem__(self, index):
        # load images
        img = cv2.imread(self.path_to_imgs[index]).astype(np.float32)
        img = torch.from_numpy(img)
        file_name = str(self.path_to_imgs[index].name)
        return img, file_name

    def __len__(self):
        return len(self.path_to_imgs)
