from torch.utils.data import Dataset
from os.path import join
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random
from torch.autograd import Variable
from sklearn.utils import shuffle
from utils.functions import pickle_load, custom_trans


class Healthy(Dataset):
    def __init__(self, root, transforms=None, mode='train', opts=None):
        self.transforms = transforms
        self.mode = mode
        self.opts = opts

        data = pickle_load(join(root, 'healthy-261-trilabel.pkl'))
        image, label = data['image'], data['label']
        image, label = shuffle(image, label, random_state=opts.random_state)

        nb_im = 30 # images per train/ val set
        if mode=='train':
            self.image = image[:-2*nb_im]
            self.label = label[:-2*nb_im]
        elif mode=='val':
            self.image = image[-2*nb_im:-nb_im]
            self.label = label[-2*nb_im:-nb_im]
        elif mode=='test':
            self.image = image[-nb_im:]
            self.label = label[-nb_im:]

    def transform(self, image, mask):
        # Random adjust_gamma
        if random.random() > 0.5:
            image = TF.adjust_gamma(image, gamma=np.random.rand()*1.3)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random shift
        if random.random() > 0.75:
            angle_rnd = np.random.randint(-15, 15)
            translate_rnd = (np.random.randint(0, 40), np.random.randint(0, 40))
            scale_rnd = np.random.randint(7, 14)*0.1
            shear_rnd = np.random.randint(-45, 45)

            image = TF.affine(image, angle=angle_rnd, translate=translate_rnd, scale=scale_rnd, shear=shear_rnd)
            mask = TF.affine(mask, angle=angle_rnd, translate=translate_rnd, scale=scale_rnd, shear=shear_rnd)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(np.asarray(mask).astype(int))
        return image, mask

    def __getitem__(self, index):
        # Should Return (unique_id, data, y)
        image, label = custom_trans(self.image[index]), custom_trans(self.label[index])
        image, label = self.transform(image, label)
        return image, label

    def __len__(self):
        return len(self.label)

    def __getsize__(self):
        return custom_trans(self.label[0]).size



def checking_data(testset):
    res_true, res_ori = [], []
    for step, (x, yt) in enumerate(testset):
        x, yt = np_var(x), np_var(yt)
        _yt, _x = var_np(yt),  var_np(x)
        res_true = (np.transpose(_yt, (0,2,3,1)))
        res_ori = (np.transpose(_x, (0,2,3,1)))
    return res_true, res_ori


def tableau(true, ori, save=False):
    l, r, c, channels = true.shape[0], true.shape[1], true.shape[2], true.shape[3]
    arr = np.zeros((r*l, c*2, channels))
    for i in range(l):
        arr[i*r:(i+1)*r, :c, :] = true[i]
        arr[i*r:(i+1)*r, c:2*c, :] = ori[i]
    arr = np.squeeze(arr)
    plt.axis('off')
    if save:
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(join('../experiments/', "data_augmentation.png"), bbox_inches = 'tight', pad_inches = 0, dpi=600)
        print('saved !')
    plt.imshow(arr)
    plt.show()



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


if __name__ == '__main__':
    root = '../data/'
    # transforms_ = transforms.Compose([transforms.RandomCrop((256,256))]  + [transforms.ToTensor()])
    testdata = Healthy(mode='test', root=root)
    print(testdata.__len__())
    print(testdata.__getsize__())
    testset = DataLoader(testdata, shuffle=False, batch_size=5)
    res_true, res_ori = checking_data(testset)
    tableau(res_true, res_ori, save=False)
