import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import sys
from torchsummary import summary

import argparse
from os.path import join
import dataloader
import pandas as pd
from pylab import savefig

from utils.functions import pickle_load, custom_trans, get_exdir, var_np, np_var, adjust_learning_rate, get_optim, save_opts
from dataloader import Healthy
from models.Unet import UNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inDir', default='../data/', type=str)
    parser.add_argument('--outDir', default= '../experiments/', type=str)
    parser.add_argument('--dataloader', default= 'skinOCT', type=str)  

    parser.add_argument('--bs', default=12, type=int)  # batchSize
    parser.add_argument('--me', default=20, type=int)  # max epoch
    parser.add_argument('--lr', default=0.001, type=float) # learning rate
    parser.add_argument('--momentum', default=0.9, type=float) # decay frequency

    parser.add_argument('--kernel_num', default=4, type=int) # number of filters
    parser.add_argument('--kernel_size', default=2**5, type=int) # size of filters
    parser.add_argument('--depth', default=5, type=int) # num of layers

    parser.add_argument('--loss', default='CE', type=str) # loss mode # CE or MSE or focal
    parser.add_argument('--weight', default=None, nargs='+', type=float) # weights
    parser.add_argument('--lr_decay', default=0.99, type=float) # decay rate
    parser.add_argument('--decay_every', default=None, type=int) # decay frequency in terms of epoch

    parser.add_argument('--optim', default='adam', type=str, choices=['adam', 'rms', 'sgd']) # decay frequency
    parser.add_argument('--random_state', default=42, type=int) # to change shuffle training/ testing/ validation
    return parser.parse_args()



def train(model, trainset, valset, testset):
    optimizer = get_optim(model, opts)
    result_val, result_test = {}, {}
    result_test['true'], result_test['pred'], result_test['ori'] = [], [], []
    result_val['true'], result_val['pred'] = [], []
    loss_train, loss_val = [], []

    for epoch in range(opts.me):

        # training
        for step, (x, yt) in enumerate(trainset):
            model.train()
            x, yt = np_var(x), np_var(yt)
            yp = model.predict(x, mode=opts.loss)
            yf = model.forward(x)
            loss = model.loss(yf, yt, mode=opts.loss, weight=opts.weight)  # BCE or CE
            optimizer.zero_grad()   # clear gradients for this training step
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
            print('step: ({},{}) | train loss: {}'.format(epoch, step,loss.data))
            loss_train.append(var_np(loss.data))

        # validation
        for step, (x, yt) in enumerate(valset):
            model.eval()
            with torch.no_grad():
                x, yt = np_var(x), np_var(yt)
                yp = model.predict(x, mode=opts.loss)
                yf = model.forward(x)
                loss = model.loss(yf, yt, mode=opts.loss, weight=opts.weight)  # BCE or CE
                print('Epoch: {} | val loss: {}'.format(epoch,loss.data))
                print("--")
        result_val['true'].append(var_np(yt))
        result_val['pred'].append(var_np(yp))
        loss_val.append(var_np(loss.data))

        # decaying lr
        if opts.decay_every is not None:
            if (epoch + 1) % opts.decay_every == 0:
                new_lr = max(opts.lr*(opts.lr_decay**((epoch + 1)//opts.decay_every)), 10**-5)
                print("new_learning rate: {:.5f}".format(new_lr))
                adjust_learning_rate(optimizer, new_lr)

    # testing
    for step, (x, yt) in enumerate(testset):
        model.eval()
        with torch.no_grad():
            x, yt = np_var(x), np_var(yt)
            yp = model.predict(x, mode=opts.loss)
            yf = model.forward(x)
            loss = model.loss(yf, yt, mode=opts.loss, weight=opts.weight)  # BCE
            print('FINAL Test loss: {}'.format(loss.data))
            result_test['true'], result_test['pred'], result_test['ori'] = var_np(yt), var_np(yp), var_np(x)

    return result_test, result_val, loss_val, loss_train


def init_normal(m):
    # try other initialisations
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)


if __name__ == '__main__':

    opts = get_args()

    # getting the structure of your model
    model = UNet(depth=opts.depth, kernel_size=opts.kernel_size, kernel_num=opts.kernel_num, n_classes=3)

    if model.useCUDA:
       model.cuda()
    # summary(model, input_size=(1, 512, 256))
    # raise
    model.apply(init_normal)

    traindata = Healthy(mode='train', root=opts.inDir, opts=opts)
    trainset = DataLoader(traindata, shuffle=True, batch_size=opts.bs)
    print('train', traindata.__len__(), traindata.__getsize__())

    valdata = Healthy(mode='val', root=opts.inDir, opts=opts)
    valset = DataLoader(valdata, shuffle=True, batch_size=opts.bs)
    print('val', valdata.__len__(), valdata.__getsize__())

    testdata = Healthy(mode='test', root=opts.inDir, opts=opts)
    testset = DataLoader(testdata, shuffle=False, batch_size=opts.bs)
    print('test', testdata.__len__(), testdata.__getsize__())


    print("begin training ...")
    result_test, result_val, loss_val, loss_train = train(model=model, trainset=trainset, valset=valset, testset=testset)
    exDir = get_exdir(opts)
    save_opts(exDir, opts)
    np.savez(join(exDir, 'result_test.npz'), true=result_test['true'], pred=result_test['pred'], ori=result_test['ori'])
    np.savez(join(exDir, 'score_val.npz'), true=result_val['true'], pred=result_val['pred'])
    np.savez(join(exDir, 'loss_val.npz'), train=loss_train, val=loss_val)
    print('Results saved in: {}'.format(exDir))
