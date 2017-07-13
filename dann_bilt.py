# coding: utf-8
# Domain Adaptation Neural Network (DANN; Ganin et al., 2017) implementation in PyTorch.
# Paper at http://www.jmlr.org/papers/volume17/15-239/source/15-239.pdf
# Author: Daniel Bartolomé Rojas (d.bartolome.r@gmail.com)
import time
import cv2
import math
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, Function

from keras.datasets import mnist, mnist_m   # mnist_m can be found in keras/ in this repo
from keras.utils import np_utils

# Gradient reversal layer (GRL) for DANN model
class GradReverse(Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -lambd)
def grad_reverse(x):
    return GradReverse()(x)

# Batch generators
def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(int(i * batch_size), int(min(size, (i + 1) * batch_size))) for i in range(0, nb_batch)]
def batch_gen(batches, id_array, data, labels):
    for batch_index, (start, end) in enumerate(batches):
        batch_ids = id_array[start:end]
        if len(batch_ids) != batch_size:
            continue
        if labels is not None:
            yield Variable(torch.from_numpy(data[batch_ids])), Variable(torch.from_numpy(labels[batch_ids]))
        else:
            yield Variable(torch.from_numpy(data[batch_ids]))          
def batch_generator(x, y, n_split):
    kf = KFold(n_splits=n_split, random_state=0)
    for i, (_, split) in enumerate(kf.split(x)):
        print('\rBatch {}/{}'.format(i+1, n_split), end='')
        x_split, y_split = x[split], y[split]
        x_split = Variable(torch.from_numpy(x_split))
        y_split = Variable(torch.from_numpy(y_split))
        yield x_split, y_split, split

# evaluate model performance
def eval_clf(x, y, n):
    out = c_clf(f_ext(Variable(torch.from_numpy(x[:n]))).view(n, -1))
    preds = out.max(1)[1]
    return accuracy_score(y_true=[np.argmax(i) for i in y[:n]], y_pred=preds.data.numpy().ravel())

# data loader for cabauw, bilt, mnist-m and mnist
def data_loader(dataset):

    # helper function to normalize images between -1 and 1
    def normalize_img(x):
        return -1 + ((x - np.min(x)) * (2)) / (np.max(x) - np.min(x))
     
    def process_img(path):
        return cv2.resize(cv2.imread(path), (28, 28), interpolation=cv2.INTER_LINEAR)
    
    if dataset == 'cabauw':
        print('Loading {} data..'.format(dataset.upper()))

        meta = pd.read_csv('./main/data/Training/ImageDescription2.csv')
        y = pd.Series(meta['vis_class']).astype('category', categories=list('ABCD')).cat.codes
        x_train = [process_img('./main/data/Training/'+name) for name in meta['basename'].values if name[0] == 'C']
        x_train = np.array(x_train, dtype=np.uint8).transpose((0,3,1,2)).astype('float32') / 255
        x_train = np.array([normalize_img(img) for img in x_train])
        y_train = np_utils.to_categorical(np.array(y, dtype=np.uint8), 4)
        
        X_train, X_val, y_train, y_val = train_test_split(x_train, y_train[2517:], test_size=0.2, random_state=43)

        return (X_train, y_train), (X_val, y_val)
    
    if dataset == 'bilt':
        print('Loading {} data..'.format(dataset.upper()))

        meta = pd.read_csv('./main/data/Training/ImageDescription2.csv')
        y = pd.Series(meta['vis_class']).astype('category', categories=list('ABCD')).cat.codes
        x_train = [process_img('./main/data/Training/'+name) for name in meta['basename'].values if name[0] == 'M']
        x_train = np.array(x_train, dtype=np.uint8).transpose((0,3,1,2)).astype('float32') / 255
        x_train = np.array([normalize_img(img) for img in x_train])
        y_train = np_utils.to_categorical(np.array(y, dtype=np.uint8), 4)
        
        X_train, X_val, y_train, y_val = train_test_split(x_train, y_train[:2517], test_size=0.2, random_state=43)

        return (X_train, y_train), (X_val, y_val)

    # load and process MNIST-M dataset
    if dataset == 'mnist_m':
        print('Loading {} data..'.format(dataset.upper()))
        
        # load mnist_m data
        mnistm = mnist_m.load_data()
        
        # load target variable from mnist
        (_, yt_train), (__, yt_test) = mnist.load_data()
        del _, __ 
        
        # create train / test split
        xt_train = mnistm[b'train']
        xt_test = mnistm[b'test']
        
        # preprocess (reshape, float32, normalize)
        xt_train = xt_train.transpose(0, 3, 1, 2).astype('float32') / 255
        xt_test = xt_test.transpose(0, 3, 1, 2).astype('float32') / 255
        
        # one hot encode target
        yt_train = np_utils.to_categorical(yt_train, 10)
        yt_test = np_utils.to_categorical(yt_test, 10)
 
        # return images
        return (xt_train[:15000], yt_train[:15000]), (xt_test[:15000], yt_test[:15000])

    # load and process MNIST dataset
    elif dataset == 'mnist':
        print('Loading {} data..'.format(dataset.upper()))
        
        # load mnist data
        (xs_train, ys_train), (xs_test, ys_test) = mnist.load_data()
        
        # preprocess (normalize, reshape, float32)
        xs_train = xs_train.reshape(xs_train.shape[0], 1, 28, 28).astype('float32') / 255
        xs_test = xs_test.reshape(xs_test.shape[0], 1, 28, 28).astype('float32') / 255
        xs_train = np.array([normalize_img(img) for img in xs_train])
        xs_test = np.array([normalize_img(img) for img in xs_test])
        
        # one hot encode target
        ys_train = np_utils.to_categorical(ys_train, 10)
        ys_test = np_utils.to_categorical(ys_test, 10)
        
        # concat MNIST images as channels to match number of MNIST-M channels
        xs_train = np.concatenate([xs_train, xs_train, xs_train], axis=1)
        xs_test = np.concatenate([xs_test, xs_test, xs_test], axis=1)
        return (xs_train, ys_train), (xs_test, ys_test)
    
    
'''
MODELS
'''
# The feature extractor inputs an image from either the source or target domain, and creates
# a lower-dimensional embedding of the image. This feature vector is then passed to the domain
# classifier or the class classifier.
class feature_extractor(nn.Module):    
    def __init__(self, n_ch):
        super(feature_extractor, self).__init__()
        self.conv1 = nn.Conv2d(n_ch, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.drop = nn.Dropout2d(.25)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))                   
        x = F.max_pool2d(F.leaky_relu(self.drop(self.conv2(x))), kernel_size=2, stride=1)
        x = F.max_pool2d(F.leaky_relu(self.bn(self.conv3(x))), kernel_size=2, stride=1)
        return x

# The domain classifier takes the feature vectors of both the source 
# and target domain to predict whether they are source or target images.
class domain_classifier(nn.Module):
    def __init__(self):
        super(domain_classifier, self).__init__()
        self.fc1 = nn.Linear(36864, 100) 
        self.fc2 = nn.Linear(100, 1)
        self.drop = nn.Dropout2d(0.25)
        
    def forward(self, x):
        x = grad_reverse(x)
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return F.sigmoid(x)

# The class classifier takes the feature vectors only for the 
# labeled source domain to predict the class of the images
class class_classifier(nn.Module):
    def __init__(self, n_classes):
        super(class_classifier, self).__init__()
        self.fc1 = nn.Linear(36864, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, n_classes)
        self.drop = nn.Dropout2d(0.25)
        
    def forward(self, x):
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = F.leaky_relu(self.drop(self.fc2(x)))
        x = self.fc3(x)
        return F.softmax(x)
    
# parameters
learning_rate = 1e-4          # learning rate
num_epochs = 15               # number of epochs to train models
num_src_epochs = 10           # number of epochs to train source-only model
batch_size = 50               # size of image sample per epoch 
source_data = 'cabauw'        # cabauw / mnist
target_data = 'bilt'          # bilt / mnist_m
input_ch = 3                  # 1 for usps, 3 for mnistm
n_classes = 4                 # mnist = 10, knmi = 4

# instantiate the models
d_clf = domain_classifier()
c_clf = class_classifier(n_classes)
f_ext = feature_extractor(input_ch)

# define loss functions
d_crit = nn.BCELoss() # binary crossentropy
c_crit = nn.CrossEntropyLoss() # categorical crossentropy

# set the optimizers
d_optimizer = optim.SGD(d_clf.parameters(), lr=learning_rate, momentum=0.9)
c_optimizer = optim.SGD(c_clf.parameters(), lr=learning_rate, momentum=0.9)
f_optimizer = optim.SGD(f_ext.parameters(), lr=learning_rate, momentum=0.9)

# load source domain dataset
(Xs_train, ys_train), (Xs_test, ys_test) = data_loader(source_data)
source_idx = np.arange(Xs_train.shape[0])
source_batches = make_batches(Xs_train.shape[0], batch_size / 2)

# load target domain dataset
(Xt_train, yt_train), (Xt_test, yt_test) = data_loader(target_data)
target_idx = np.arange(Xt_train.shape[0])
target_batches = make_batches(Xt_train.shape[0], batch_size / 2)

# init necessary objects
num_steps = num_epochs * (Xs_train.shape[0] / batch_size)
yd = Variable(torch.from_numpy(np.vstack([np.repeat(1, int(batch_size / 2)), np.repeat(0, int(batch_size / 2))])))
j = 0

# pre-train source only model
print('\nPre-training source-only model..')
for i in range(num_src_epochs):
    start_time = time.time()
    source_gen = batch_gen(source_batches, source_idx, Xs_train, ys_train)
    
    # iterate over batches
    for (xs, ys) in source_gen:
        
        # exit when batch size mismatch
        if len(xs) != batch_size / 2:
            continue
        
        # reset gradients
        f_ext.zero_grad()
        c_clf.zero_grad()
        
        # calculate class_classifier predictions
        c_out = c_clf(f_ext(xs).view(int(batch_size / 2), -1))
        
        # optimize feature_extractor and class_classifier with output
        f_c_loss = c_crit(c_out, ys.float())
        f_c_loss.backward(retain_variables = True)
        c_optimizer.step()
        f_optimizer.step()
        
        # print batch statistics
        time.sleep(1)
        print('\rEpoch {} - {}s - loss: {}'.format(
            i+1, round(time.time() - start_time),
            format(f_c_loss.data[0], '.4f')), end='')
    
    # print epoch statistics
    s_acc = []
    _n_split = int(round(Xs_test[:450].shape[0] / batch_size))
    _gen = batch_generator(Xs_test[:450], ys_test[:450], _n_split)
    for xs in _gen:
        pred = c_clf(f_ext(xs[0]).view(batch_size, -1))
        s_acc.append(pred.max(1)[1].data.numpy().ravel())
    s_acc = accuracy_score(y_true=[np.argmax(i) for i in ys_test[:450]], y_pred=[x for y in s_acc for x in y])
    print(' - val_acc: {}'.format(format(s_acc, '.4f')))


# print target accuracy with source model
t_acc = []
_n_split = int(round(Xt_test[:450].shape[0] / batch_size))
_target_gen = batch_generator(Xt_test[:450], yt_test[:450], _n_split)
for xt in _target_gen:
    pred = c_clf(f_ext(xt[0]).view(batch_size, -1))
    t_acc.append(pred.max(1)[1].data.numpy().ravel())
t_acc = accuracy_score(y_true=[np.argmax(i) for i in yt_test[:450]], y_pred=[x for y in t_acc for x in y])
print('\nTarget accuracy with source model: {}\n'.format(format(t_acc, '.4f')))

# train DANN model
print('Training DANN model..')
for i in range(num_epochs):
    start_time = time.time()
    source_gen = batch_gen(source_batches, source_idx, Xs_train, ys_train)
    target_gen = batch_gen(target_batches, target_idx, Xt_train, None)

    # iterate over batches
    for (xs, ys) in source_gen:
        
        # update lambda and learning rate as suggested in the paper
        p = float(j) / num_steps
        lambd = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.01 / (1. + 10 * p)**0.75
        d_optimizer.lr = lr
        c_optimizer.lr = lr
        f_optimizer.lr = lr
        
        # exit if batch size incorrect, get next target batch
        if len(xs) != batch_size / 2:
            continue
        xt = next(target_gen)
        
        # concatenate source and target batch
        x = torch.cat([xs, xt.float()], 0)
        
        # 1) train feature_extractor and class_classifier on source batch
        # reset gradients
        f_ext.zero_grad()
        c_clf.zero_grad()
        
        # calculate class_classifier predictions on batch xs
        c_out = c_clf(f_ext(xs).view(int(batch_size / 2), -1))
        
        # optimize feature_extractor and class_classifier on output
        f_c_loss = c_crit(c_out, ys.float())
        f_c_loss.backward(retain_variables = True)
        c_optimizer.step()
        f_optimizer.step()
        
        # 2) train feature_extractor and domain_classifier on full batch x
        # reset gradients
        f_ext.zero_grad()
        d_clf.zero_grad()
        
        # calculate domain_classifier predictions on batch x
        d_out = d_clf(f_ext(x).view(batch_size, -1))
        
        # optimize feature_extractor and domain_classifier on output
        f_d_loss = d_crit(d_out, yd.float())
        f_d_loss.backward(retain_variables = True)
        d_optimizer.step()
        f_optimizer.step()
        
        print('s')
        # print batch statistics
        print('\rEpoch {} - {}s - domain_loss: {} - class_loss: {}'.format(
            i+1, round(time.time() - start_time),
            format(f_d_loss.data[0], '.4f'),
            format(f_c_loss.data[0], '.4f')), end='')
        print('s')
    
    # print epoch statistics
    s_acc = []
    _n_split = int(round(Xs_test[:450].shape[0] / batch_size))
    _gen = batch_generator(Xs_test[:450], ys_test[:450], _n_split)
    for xs in _gen:
        pred = c_clf(f_ext(xs[0]).view(batch_size, -1))
        s_acc.append(pred.max(1)[1].data.numpy().ravel())
    s_acc = accuracy_score(y_true=[np.argmax(i) for i in ys_test[:450]], y_pred=[x for y in s_acc for x in y])

    t_acc = []
    _n_split = int(round(Xt_test[:450].shape[0] / batch_size))
    _target_gen = batch_generator(Xt_test[:450], yt_test[:450], _n_split)
    for xt in _target_gen:
        pred = c_clf(f_ext(xt[0]).view(batch_size, -1))
        t_acc.append(pred.max(1)[1].data.numpy().ravel())
    t_acc = accuracy_score(y_true=[np.argmax(i) for i in yt_test[:450]], y_pred=[x for y in t_acc for x in y])

    print(' - t_acc: {} - s_acc: {}'.format(format(t_acc, '.4f'), format(s_acc, '.4f')))

# create predictions with fake target images
t_acc = []
_n_split = int(round(Xt_test[:450].shape[0] / batch_size))
_target_gen = batch_generator(Xt_test[:450], yt_test[:450], _n_split)
for xt in _target_gen:
    pred = c_clf(f_ext(xt[0]).view(batch_size, -1))
    t_acc.append(pred.max(1)[1].data.numpy().ravel())
t_acc = accuracy_score(y_true=[np.argmax(i) for i in yt_test[:450]], y_pred=[x for y in t_acc for x in y])

# return accuracy and confusion matrix 
print('\n{} test accuracy: {}'.format(target_data.upper(), round(t_acc, 3)))