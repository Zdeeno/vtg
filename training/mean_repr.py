#!/usr/bin/env python3.9
import copy

import numpy as np
import torch as t
from parser_data import SingleReprDataset
from model import Siamese, get_custom_CNN, save_model, load_model, jit_save, jit_load, get_parametrized_model, CNNHead
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW, Adam, RMSprop
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from torch.nn.functional import softmax
from tqdm import tqdm
import argparse
from torch.nn.functional import conv2d, conv1d
import time
import matplotlib.pyplot as plt
import os

device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

OUT_NAME = "gen_strom_forest"
PATH = "/home/robot/.ros"
SUBFOLDS = ["strom22", "strom23"]
FLIPS = [0, 0]

EPOCHS = 1
EVAL_RATE = 10
BATCH_SIZE = 10
REPR_SIZE = (1, 16, 6, 64)
LR = 0.003
PAD = 32

dataset = SingleReprDataset(path=PATH,
                            subfolder_list=SUBFOLDS,
                            flip_list=FLIPS)
train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)


def match_corr(embed_ref, embed_srch, padding=None):
    """ Matches the two embeddings using the correlation layer. As per usual
    it expects input tensors of the form [B, C, H, W].
    Args:
        embed_ref: (torch.Tensor) The embedding of the reference image, or
            the template of reference (the average of many embeddings for
            example).
        embed_srch: (torch.Tensor) The embedding of the search image.
    Returns:
        match_map: (torch.Tensor) The correlation between
    """
    b, c, h, w = embed_srch.shape
    _, _, h_ref, w_ref = embed_ref.shape
    # Here the correlation layer is implemented using a trick with the
    # conv2d function using groups in order to do the correlation with
    # batch dimension. Basically we concatenate each element of the batch
    # in the channel dimension for the search image (making it
    # [1 x (B.C) x H' x W']) and setting the number of groups to the size of
    # the batch. This grouped convolution/correlation is equivalent to a
    # correlation between the two images, though it is not obvious.

    match_map = conv2d(embed_srch.view(1, b * c, h, w), embed_ref, groups=b, padding=(0, PAD))
    match_map = match_map.permute(1, 0, 2, 3)
    return match_map

class LearnedRepr(t.nn.Module):
    def __init__(self, init_tensor=None):
        super().__init__()
        starting_repr = t.normal(0.5, 0.01, REPR_SIZE)
        # starting_repr = t.ones((1, 3, 384, 512))
        if init_tensor is None:
            self.trained_repr = t.nn.Parameter(starting_repr, requires_grad=True).float()
        else:
            self.trained_repr = t.nn.Parameter(init_tensor, requires_grad=True).float()

    def forward(self, batch_size):
        x = self.trained_repr.repeat((batch_size, 1, 1, 1))
        return x


init_tensor = t.zeros(REPR_SIZE).to(device)
rounds = 0
for batch in tqdm(train_loader):
    tmp = batch.to(device)
    init_tensor += t.sum(tmp, dim=0)
    rounds += tmp.shape[0]
init_tensor /= rounds


# trained_repr = LearnedRepr(init_tensor=init_tensor).to(device)
trained_repr = LearnedRepr(init_tensor=init_tensor).to(device)
last_repr = trained_repr.trained_repr[0].detach()
optimizer = Adam(trained_repr.parameters(), lr=LR)
# loss = CrossEntropyLoss()
loss = BCEWithLogitsLoss()


def train_loop(epoch):
    trained_repr.train()
    loss_sum = 0
    print("Training model epoch", epoch)
    round_num = 0
    for batch in tqdm(train_loader):
        batch_reprs = batch.to(device)

        target = t.zeros(batch_reprs.shape[0], 65, device=device).float()
        target[:, 32] = 1.0
        target[:, 33] = 0.66
        target[:, 31] = 0.66
        target[:, 34] = 0.33
        target[:, 30] = 0.33

        optimizer.zero_grad()
        match_out = match_corr(trained_repr(batch_reprs.shape[0]), batch_reprs.float())
        l = loss(match_out.squeeze(1).squeeze(1), target)
        l.backward()

        optimizer.step()
        round_num += 1
        loss_sum += l.cpu().detach().numpy()


    print("Training of epoch", epoch, "ended with loss", loss_sum / len(train_loader))


def eval_loop(epoch):
    # TODO: continue here
    global last_repr
    last_repr = trained_repr.trained_repr.detach()
    match_sum = np.zeros(65)
    iter = 0
    for batch in tqdm(train_loader):
        batch_reprs = batch.to(device)
        match_out = match_corr(last_repr.repeat((batch_reprs.shape[0], 1, 1, 1)), batch_reprs.float(),
                               padding=PAD).squeeze(1).squeeze(1)
        match_sum += np.sum(match_out.cpu().numpy(), axis=0)
        iter += batch_reprs.shape[0]
    ret = match_sum/iter
    plt.plot(ret)
    plt.grid()
    plt.show()
    return 0.0


out_path = "/home/robot/.ros/" + OUT_NAME
os.mkdir(out_path)
out_repr = trained_repr.trained_repr.detach().numpy()[0]
struct_save = {"representation": out_repr}
with open(out_path + "/1000.0.npy", 'wb') as fp:
    np.save(fp, struct_save, fix_imports=False)

