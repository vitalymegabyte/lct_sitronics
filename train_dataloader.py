import os
import threading
import time

from tqdm import tqdm
from convert import read_img
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.half if torch.cuda.is_available() else torch.float32

def parse_input(x):
    with torch.no_grad():
        if len(x.shape) == 3:
            x = x[None, ...]

        if isinstance(x, np.ndarray):
            x = torch.tensor(x).permute(0, 3, 1, 2) / 255

        while x.shape[2] > 2000:
            x = x[:, :, ::2, ::2]

        x = x[:, :, x.shape[2] % 4 :, x.shape[3] % 4 :]

        return x.to(device, dtype=dtype)

class Loader:
    def __init__(self):
        images = os.listdir("train_dataset/maps")
        images.sort()
        self.images_bing = images[: len(images) // 2]
        self.images_google = images[len(images) // 2 :]

        self.images_bing = [parse_input(
                read_img(os.path.join("train_dataset/maps/", img))["image"]
            ) for img in tqdm(self.images_bing, desc="loading bing")]
        self.images_google = [parse_input(
                read_img(os.path.join("train_dataset/maps/", img))["image"]
            ) for img in tqdm(self.images_google, desc="loading google")]

    def __len__(self):
        return len(self.images_bing)