import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import argparse
import os
from os.path import join, exists
from scipy.ndimage.filters import convolve
import torch
from torch.autograd import Variable
import json


def save_to_json(data, exDir, filename, indent=1):
    with open(join(exDir, filename), 'w') as fw:
        json.dump(data, fw, indent=indent)


def save_opts(exDir, opts):
    # save the input args to
    save_to_json(opts.__dict__, exDir, 'opts.json')


def pickle_load(file_path):
    with open(file_path, 'rb') as fr:
        data = pickle.load(fr)
    return data


def custom_trans(img):
    width, height = img.size
    img = img.crop((420,0,width-400,height))
    img = img.resize((512, 256), Image.ANTIALIAS)
    # img = img.resize((int(0.5*img.size[0]), int(0.5*img.size[1])), Image.ANTIALIAS)
    if img.mode == 'L':
        return img.convert('L')
    else:
        return img


def get_exdir(opts):
    outDir = opts.outDir
    mainDir = join(outDir)
    experiment_index = 0
    exDir = join(mainDir, 'exp_{}'.format(experiment_index))
    while exists(exDir):
        experiment_index += 1
        exDir = join(mainDir, 'exp_{}'.format(experiment_index))
    os.makedirs(exDir)
    return exDir

def var_np(x):
    # variable to numpy
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def np_var(x):
    # numpy to var
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def adjust_learning_rate(optimizer, new_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def get_optim(model, opts, lr=None):
    if lr is None:
        lr = opts.lr
    if opts.optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opts.optim == 'rms':
        return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=opts.momentum)
    else:
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=opts.momentum)
