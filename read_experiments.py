import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from sys import argv
import argparse
from sklearn.metrics import confusion_matrix, jaccard_score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataloader', default='Healthy', type=str)
    parser.add_argument('--model', default='UNet', type=str)
    parser.add_argument('--loss', default='BCE', type=str)
    parser.add_argument('--nb', default='0', type=str) # number experimnets
    # parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--outDir', default= '../experiments/', type=str)
    return parser.parse_args()


def load_result_test(opts):
    exp_path = os.path.join(opts.outDir, opts.dataloader, opts.model, opts.loss, 'exp_{}'.format(opts.nb), 'result_test.npz')
    res = np.load(exp_path)
    pred, true, ori = res['pred'], res['true'], res['ori']
    pred = 1*(pred>0.5)
    if pred.shape[1]==3:
        pred_ = np.zeros((true.shape))
        pred_[:,0,:,:] = 0*pred[:,0,:,:] + 1*pred[:,1,:,:] + 2*pred[:,2,:,:]
        pred = pred_
    print(pred_.shape)
    print(pred.shape, true.shape, ori.shape)
    print(np.unique(pred), np.unique(true))
    pred, true, ori = np.transpose(pred, (0,2,3,1)), np.transpose(true, (0,2,3,1)), np.transpose(ori, (0,2,3,1))
    return pred[:5], true[:5], ori[:5]


def tableau(pred, true, ori, save=False, threshold=False):
    print( pred.shape, true.shape, ori.shape)
    r, c, l, channels = pred.shape[1], pred.shape[2], pred.shape[0], pred.shape[3]
    arr = np.zeros((r*l, c*3, channels))
    # if threshold:
    #     arr = np.zeros((r*l, c*4, channels))

    for i in range(l):
        arr[i*r:(i+1)*r, :c, :] = true[i]
        arr[i*r:(i+1)*r, c:2*c, :] = pred[i]
        arr[i*r:(i+1)*r, 2*c:3*c, :] = ori[i]
        if threshold :
            thre = 0.5*(np.max(pred[0]) - np.min(pred[0]))
            arr[i*r:(i+1)*r, 3*c:, :] = pred[i] > thre
    arr = np.squeeze(arr)
    # f = plt.figure()
    plt.axis('off')
    if save:
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.savefig(join('../experiments/', "filters.png"), bbox_inches = 'tight', pad_inches = 0, dpi=600)
        print('saved !')
    plt.imshow(arr)


def load_loss(opts):
    exp_folder = os.path.join(opts.outDir, opts.dataloader, opts.model, opts.loss, 'exp_{}'.format(opts.nb))
    loss_ = np.load(os.path.join(exp_folder, 'loss_val.npz'))
    loss_val, loss_train = loss_['val'], loss_['train']
    me = len(loss_val)
    xaxis_val, loss_train = np.linspace(0, 12*me, me+1)[:me], loss_train[:12*me]
    fig, ax = plt.subplots()
    ax.plot(xaxis_val, loss_val, 'go', label='validation')
    ax.plot(loss_train, 'orange', label='training')
    legend = ax.legend(loc='center right', fontsize='small')
    plt.ylabel(opts.loss)
    plt.xlabel('# batchs')
    plt.ylim(0,1.4)
    plt.title('Loss value over training - {}'.format(opts.model))
    plt.savefig(os.path.join(exp_folder, 'loss.png'))
    plt.close()


def load_score(opts, threshold=0.5):
    exp_folder = os.path.join(opts.outDir, opts.dataloader, opts.model, opts.loss, 'exp_{}'.format(opts.nb))
    score_val = np.load(os.path.join(exp_folder, 'score_val.npz'))
    yt, yp = np.squeeze(score_val['true']), np.squeeze(score_val['pred'])
    yp = (yp > threshold)*1.
    average='binary'
    if yp.shape[2]==3:
        yp_ = 0*yp[:,:,0,:,:] + 1*yp[:,:,1,:,:] + 2*yp[:,:,2,:,:]
        yp = yp_
        average='weighted'

    metrics = {}
    metrics['jaccard'], metrics['dice'] = [], []
    for epoch in range(yt.shape[0]):
        _yt, _yp = yt[epoch], yp[epoch]
        jaccard = jaccard_score(_yt.flatten(), _yp.flatten(), average=average)
        metrics['jaccard'].append(jaccard)
        metrics['dice'].append(2*jaccard/(jaccard+1))

    fig, ax = plt.subplots()
    ax.plot(metrics['jaccard'], label='jaccard index')
    ax.plot(metrics['dice'], label='dice coefficient')
    # ax.plot(metrics['jaccard'], label='jaccard index')
    # ax.plot(metrics['jaccard'], label='jaccard index')
    legend = ax.legend(loc='center right', fontsize='small')
    plt.ylabel(opts.loss)
    plt.xlabel('# epoch')
    plt.title('Metrics over training - {}'.format(opts.model))
    plt.savefig(os.path.join(exp_folder, 'score.png'))
    plt.close()


def print_conf_mat(y_test, y_pred, class_name):
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = pd.DataFrame(cm_norm, columns=class_name, index=class_name)
    cm_norm = cm_norm.round(2)
    print(cm_norm.to_string())



def load_metrics(opts, threshold=0.5):
    exp_folder = os.path.join(opts.outDir, opts.dataloader, opts.model, opts.loss, 'exp_{}'.format(opts.nb))
    score_val = np.load(os.path.join(exp_folder, 'result_test.npz'))
    yt, yp = np.squeeze(score_val['true']), np.squeeze(score_val['pred'])
    yp = (yp > threshold)*1.
    if yp.shape[1]==3:
        yp_ = 0*yp[:,0,:,:] + 1*yp[:,1,:,:] + 2*yp[:,2,:,:]
        yp = yp_

    print_conf_mat(yt.flatten(), yp.flatten(), class_name=['background', 'epidermis', 'dermis'])


if __name__ == '__main__':
    opts = get_args()

    load_metrics(opts)
    load_score(opts)
    load_loss(opts)

    pred, true, ori = load_result_test(opts)
    arr = tableau(pred, true, ori)

    exp_folder = os.path.join(opts.outDir, opts.dataloader, opts.model, opts.loss, 'exp_{}'.format(opts.nb))
    plt.savefig(os.path.join(exp_folder, 'tableau.png'))
