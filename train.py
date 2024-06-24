import os
import time
import numpy as np
import torch
import cv2 as cv
from convert import read_img
from modules.model import XFeatModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train_dataloader import Loader

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.half if torch.cuda.is_available() else torch.float32

weights = os.path.abspath(os.path.dirname(__file__)) + '/weights/xfeat.pt'

model = XFeatModel().to(device, dtype=dtype)
model.load_state_dict(torch.load(weights))

sample_model = XFeatModel().to(device, dtype=dtype)
sample_model.load_state_dict(torch.load(weights))

model.train()
sample_model.eval()

img = read_img("train_dataset/maps/b1.tif")["image"]

additional_loss_fn = torch.nn.L1Loss()
# Инициализация SummaryWriter
writer = SummaryWriter('logs/experiment')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

train_loader = Loader()


try:
    i = 0
    for epoch in tqdm(range(10)):
        for img_b, img_g in tqdm(zip(train_loader.images_bing, train_loader.images_google)):

            with torch.no_grad():
                img_b_sample = sample_model(img_b)
                img_g_sample = sample_model(img_g)
            begin_time = time.time()

            img_b_model = model(img_b)
            img_g_model = model(img_g)

            feat_loss = -torch.nn.functional.cosine_similarity(
                img_b_model[0], img_g_model[0], dim=1
            ).mean()

            keypoints_loss = additional_loss_fn(
                img_b_model[1], img_b_sample[1]
            ) + additional_loss_fn(img_g_model[1], img_g_sample[1])
            heatmap_loss = additional_loss_fn(
                img_b_model[2], img_b_sample[2]
            ) + additional_loss_fn(img_g_model[2], img_g_sample[2])

            loss = feat_loss + keypoints_loss + heatmap_loss

            loss.backward()
            optimizer.step()

            writer.add_scalar('training_loss', loss, i)
            writer.flush()
            i += 1
        torch.save(model, f"checkpoints/epoch_{epoch}.pt")
except KeyboardInterrupt:
    pass
writer.close()

cv.imwrite("out.png", img)
