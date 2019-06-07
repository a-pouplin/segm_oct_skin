# code modified from : https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py

# Args:
#     in_channel (int): number of input channels
#     n_classes (int): number of output channels
#     depth (int): depth of the network
#     kernel_size (int): number of filters in the first layer is 2**kernel_size
#     padding (bool): if True, apply padding such that the input shape is the same as the output. This may introduce artifacts
#     batch_norm (bool): Use BatchNorm after layers with an activation function
#     up_mode (str): one of 'upconv' or 'upsample'. 'upconv' will use transposed convolutions for learned upsampling. 'upsample' will use bilinear upsampling.

import torch
from torch import nn
import torch.nn.functional as F
from utils.functions import var_np, np_var


class Model(nn.Module):
    def __init__(self, in_channel=1, n_classes=1, depth=5, kernel_size=2**5, kernel_num=4, padding=False, batch_norm=False, up_mode='upconv'):
        super(Model, self).__init__()
        self.kernel_size = kernel_size # size of kernels
        self.kernel_num = kernel_num
        self.depth = depth
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.padding = padding
        self.batch_norm = batch_norm
        self.up_mode = up_mode
        self.useCUDA = torch.cuda.is_available()

    def forward(self, x):
        # forward with non linear layers
        raise NotImplementedError

    def predict(self, x, mode=None):
        x = self.forward(x)
        return self.softmax(x)

    def loss(self, yf, yt, weight=None, mode=None):
        # mutli label target
        if mode=='CE':
            yt = yt.squeeze()
            def _loss(yf, yt):
                criterion =  nn.CrossEntropyLoss() # nn.LogSoftmax() + nn.NLLloss()
                return criterion(yf, yt)
        elif mode == 'MSE': # softmax + MSE
            yf = self.softmax(yf)
            _loss = nn.MSELoss()

        # need to implement dice loss
        loss = _loss(yf, yt)
        return loss


class UNet(Model):
    def __init__(self, in_channel=1, n_classes=1, depth=5, kernel_size=6, kernel_num=4, padding=False, batch_norm=False, up_mode='upconv'):
        super(UNet, self).__init__(in_channel=1, n_classes=1, depth=5, kernel_size=6, kernel_num=4, padding=False,
                                    batch_norm=False, up_mode='upconv')
        print('UNet model')
        self.batch_norm = batch_norm
        self.up_mode = up_mode
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        prev_channels = in_channel
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, kernel_num*(2**i), padding, batch_norm, kernel_size))
            prev_channels = kernel_num*(2**i)
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, kernel_num*(2**i), up_mode, padding, batch_norm, kernel_size))
            prev_channels = kernel_num*(2**i)
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
        self.sigm = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])
        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, kernel_size):
        super(UNetConvBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size, kernel_size=kernel_size+1, stride=1, padding=int(kernel_size/2), dilation=1, groups=1, bias=True))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.Conv2d(out_size, out_size, kernel_size=kernel_size+1, stride=1, padding=int(kernel_size/2), dilation=1, groups=1, bias=True))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, kernel_size):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), nn.Conv2d(in_size, out_size, kernel_size=1))
        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm, kernel_size)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out
