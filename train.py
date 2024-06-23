import os
import numpy as np
import torch
import cv2 as cv
from convert import read_img
from modules.model import XFeatModel
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

weights = os.path.abspath(os.path.dirname(__file__)) + '/weights/xfeat.pt'

model = XFeatModel().to(device)
model.load_state_dict(torch.load(weights))

sample_model = XFeatModel().to(device)
sample_model.load_state_dict(torch.load(weights))

model.train()
sample_model.eval()

img = read_img("train_dataset/maps/b1.tif")["image"]


def parse_input(x):
    if len(x.shape) == 3:
        x = x[None, ...]

    if isinstance(x, np.ndarray):
        x = torch.tensor(x).permute(0, 3, 1, 2) / 255

    while x.shape[2] > 5000:
        x = x[:, :, ::2, ::2]

    x = x[:, :, x.shape[2] % 4 :, x.shape[3] % 4 :]

    return x.to(device)


images = os.listdir("train_dataset/maps")
images.sort()
images_bing = images[: len(images) // 2]
images_google = images[len(images) // 2 :]

additional_loss_fn = torch.nn.MSELoss()
# Инициализация SummaryWriter
writer = SummaryWriter('logs/experiment')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

try:
    for epoch in range(10):
        for i, (img_b, img_g) in enumerate(zip(images_bing, images_google)):
            img_b = parse_input(
                read_img(os.path.join("train_dataset/maps/", img_b))["image"]
            )
            img_g = parse_input(
                read_img(os.path.join("train_dataset/maps/", img_g))["image"]
            )

            with torch.no_grad():
                img_b_sample = sample_model(img_b)
                img_g_sample = sample_model(img_g)

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
        torch.save(model, f"checkpoints/epoch_{i}.pt")
except KeyboardInterrupt:
    pass
writer.close()

cv.imwrite("out.png", img)
