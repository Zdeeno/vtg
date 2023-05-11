#!/usr/bin/env python3.9
import copy

import numpy as np
import torch as t
from parser_data import FlippedPairDataset, SingleFlippedDataset
from model import Siamese, get_custom_CNN, save_model, load_model, jit_save, jit_load, get_parametrized_model, CNNHead
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW, Adam, RMSprop
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from torch.nn.functional import softmax
from tqdm import tqdm
from utils import plot_samples, batch_augmentations, plot_samples_lowdim, show_img
import argparse
# import wandb
import time


def get_pad(crop):
    return (crop - 8) // 16


VISUALISE = True
WANDB = False
NAME = "generalize"
BATCH_SIZE = 24  # higher better
EPOCHS = 32
LR = 1.5
EVAL_RATE = 1
CROP_SIZES = [40]  # [56 + 16*i for i in range(5)]
FRACTION = 8
PAD = 32
SMOOTHNESS = 2
NEGATIVE_FRAC = 0.0
LAYER_POOL = True
FILTER_SIZE = 3
EMB_CHANNELS = 128
RESIDUAL = 2

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
# device = t.device("cpu")
batch_augmentations = batch_augmentations.to(device)
print(CROP_SIZES)


parser = argparse.ArgumentParser(description='Training script for image alignment.')
parser.add_argument('--nm', type=str, help="name of the model", default=NAME)
parser.add_argument('--lr', type=float, help="learning rate", default=LR)
parser.add_argument('--bs', type=int, help="batch size", default=BATCH_SIZE)
parser.add_argument('--nf', type=float, help="negative fraction", default=NEGATIVE_FRAC)
# parser.add_argument('--dev', type=str, help="device", required=True)
parser.add_argument('--lp', type=bool, help="layer pooling", default=LAYER_POOL)
parser.add_argument('--fs', type=int, help="filter size", default=FILTER_SIZE)
parser.add_argument('--ech', type=int, help="embedding channels", default=EMB_CHANNELS)
parser.add_argument('--sm', type=int, help="smoothness of target", default=SMOOTHNESS)
parser.add_argument('--cs', type=int, help="crop size", default=CROP_SIZES[0])
parser.add_argument('--res', type=int, help="0 - no residual, 1 - residual 2 - S&E layer, 3 - both", default=RESIDUAL)


args = parser.parse_args()

print("Argument values: \n", args)
NAME = args.nm
LR = 10**-args.lr
BATCH_SIZE = args.bs
NEGATIVE_FRAC = args.nf
# device = args.dev
LAYER_POOL = args.lp
FILTER_SIZE = args.fs
EMB_CHANNELS = args.ech
CROP_SIZES[0] = args.cs
SMOOTHNESS = args.sm
assert args.res in [0, 1, 2, 3], "Residual type is wrong"
RESIDUAL = args.res
EVAL_PATH = "/home/zdeeno/Documents/Datasets/grief_jpg"


dataset = SingleFlippedDataset()
# dataset = CompoundDataset(CROP_SIZES[0], FRACTION, SMOOTHNESS,
#                           nordland_path="/home/zdeeno/Documents/Datasets/nordland_rectified",
#                           eu_path="/home/zdeeno/Documents/Datasets/eulongterm_rectified")
# dataset = RectifiedEULongterm(CROP_SIZES[0], FRACTION, SMOOTHNESS,
#                               path="/home/zdeeno/Documents/Datasets/eulongterm_rectified")
train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)

# backbone = get_pretrained_VGG11()   # use pretrained network - PAD = 7
# backbone = get_custom_CNN(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS, LAYER_NORM, RESIDUAL)
# backbone = get_UNet()
# model = Siamese(backbone, padding=PAD, eb=END_BN).to(device)
model = get_parametrized_model(LAYER_POOL, FILTER_SIZE, EMB_CHANNELS, RESIDUAL, PAD, device)


class LearnedImg(t.nn.Module):
    def __init__(self, init_tensor=None):
        super().__init__()
        starting_repr = t.normal(0.5, 0.01, (1, 3, 384, 512))
        # starting_repr = t.ones((1, 3, 384, 512))
        if init_tensor is None:
            self.trained_repr = t.nn.Parameter(starting_repr, requires_grad=True)
        else:
            self.trained_repr = t.nn.Parameter(init_tensor, requires_grad=True)

    def forward(self, batch_size):
        x = self.trained_repr.repeat((batch_size, 1, 1, 1))
        return x


init_tensor = t.zeros((1, 3, 384, 512)).to(device)
rounds = 0
for batch in tqdm(train_loader):
    imgs = batch.to(device)
    init_tensor += t.mean(imgs, dim=0)
    rounds += 1
init_tensor /= rounds


trained_repr = LearnedImg(init_tensor=None).to(device)
last_repr = trained_repr.trained_repr[0].detach()
optimizer = AdamW(model.parameters(), lr=LR)
optimizer2 = Adam(trained_repr.parameters(), lr=LR)
# loss = CrossEntropyLoss()
loss = BCEWithLogitsLoss()
in_example = (t.zeros((1, 3, 512, 512)).to(device).float(), t.zeros((1, 3, 512, 512)).to(device).float())


def hard_negatives(batch, heatmaps):
    if batch.shape[0] == BATCH_SIZE - 1:
        num = int(BATCH_SIZE * NEGATIVE_FRAC)
        if num % 2 == 1:
            num -= 1
        indices = t.tensor(np.random.choice(np.arange(0, BATCH_SIZE), num), device=device)
        heatmaps[indices, :] = 0.0
        tmp_t = t.clone(batch[indices[:num//2]])
        batch[indices[:num//2]] = batch[indices[num//2:]]
        batch[indices[num//2:]] = tmp_t
        return batch, heatmaps
    else:
        return batch, heatmaps


def train_loop(epoch):
    PAD = get_pad(CROP_SIZES[0])
    model.train()
    trained_repr.train()
    loss_sum = 0
    print("Training model epoch", epoch)
    round_num = 0
    for batch in tqdm(train_loader):
        imgs = batch.to(device)
        # create target
        heatmap = t.zeros(imgs.shape[0], 65, device=device)
        heatmap[:, 32] = 1.0
        heatmap[:, 33] = 0.66
        heatmap[:, 31] = 0.66
        heatmap[:, 34] = 0.33
        heatmap[:, 30] = 0.33
        # heatmap[:, 35] = 0.25..
        # heatmap[:, 29] = 0.25

        # source = batch_augmentations(source)
        # if NEGATIVE_FRAC > 0.01:
        #     batch, heatmap = hard_negatives(source, heatmap)
        # target = batch_augmentations(target)
        shift = np.random.randint(-12, 12)
        target = t.roll(imgs, shifts=shift * 8, dims=-1)
        out = model(trained_repr(imgs.shape[0]), target, padding=32)
        heatmap = t.roll(heatmap, shifts=-shift, dims=-1)

        optimizer.zero_grad()
        optimizer2.zero_grad()
        l = loss(out, heatmap)
        loss_sum += l.cpu().detach().numpy()
        l.backward()
        # print(trained_repr.trained_repr.grad)
        # optimizer.step()
        optimizer2.step()
        round_num += 1
        # if round_num > 20:
        #     break
        #plot_samples(source[0].cpu(),
        #             target[0].cpu(),
        #             heatmap[0].cpu(),
        #             name=str(round_num),
        #             dir="viz_" + NAME + "/" + str(epoch) + "/")


    print("Training of epoch", epoch, "ended with loss", loss_sum / len(train_loader))


def eval_loop(epoch):
    global last_repr
    print("Validating model after epoch", epoch)
    show_img(trained_repr.trained_repr[0].detach().cpu())

    print("repr diff:", t.sum(trained_repr.trained_repr[0].detach() - last_repr))
    print("repr sum:", t.sum(trained_repr.trained_repr[0].detach()))

    last_repr = trained_repr.trained_repr[0].detach()
    return 0.0


LOAD_EPOCH = 0
model, optimizer = load_model(model, "model/model_eunord.pt", optimizer=optimizer)


lowest_err = 0
best_model = None

for epoch in range(LOAD_EPOCH, EPOCHS):
    if epoch % EVAL_RATE == 0:
        err = eval_loop(epoch)
        # if err < lowest_err:
        #     lowest_err = err
        #     best_model = copy.deepcopy(model)
        #     save_model(model, NAME, err, optimizer)

    train_loop(epoch)
