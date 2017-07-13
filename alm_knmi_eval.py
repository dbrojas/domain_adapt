# coding: utf-8
# Approximate Label Matching (ALM; Ash et al., 2017) implementation in PyTorch.
# Read the paper at https://arxiv.org/pdf/1602.04889.pdf
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split

import cv2
import matplotlib.pyplot as plt
from pylab import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from keras.datasets import mnist , mnist_m
from keras.utils import np_utils

# Batch generators
def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(int(i * batch_size), int(min(size, (i + 1) * batch_size))) for i in range(0, nb_batch)]
def batch_gen(batches, id_array, data, labels):
    for batch_index, (start, end) in enumerate(batches):
        print('\rBatch {}/{}'.format(batch_index+1, len(batches)), end='')
        batch_ids = id_array[start:end]
        if labels is not None:
            yield Variable(torch.from_numpy(data[batch_ids])), Variable(torch.from_numpy(labels[batch_ids])), batch_ids
        else:
            yield Variable(torch.from_numpy(data[batch_ids])), batch_ids
def batch_generator(x, y, n_split):
    kf = KFold(n_splits=n_split, random_state=0)
    for i, (_, split) in enumerate(kf.split(x)):
        print('\rBatch {}/{}'.format(i+1, n_split), end='')
        x_split, y_split = x[split], y[split]
        x_split = Variable(torch.from_numpy(x_split))
        y_split = Variable(torch.from_numpy(y_split))
        yield x_split, y_split, split
    
# data loader for cabauw, bilt, mnist-m and mnist
def data_loader(dataset):

    # helper function to normalize images between -1 and 1
    def normalize_img(x):
        return -1 + ((x - np.min(x)) * (2)) / (np.max(x) - np.min(x))
    
    def process_img(path):
        img = cv2.resize(cv2.imread(path), (28, 28), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
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
        return (xt_train, yt_train), (xt_test, yt_test)

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
        
        # concat MNIST images depthwise to match number of MNIST-M channels
        xs_train = np.concatenate([xs_train, xs_train, xs_train], axis=1)
        xs_test = np.concatenate([xs_test, xs_test, xs_test], axis=1)
        return (xs_train, ys_train), (xs_test, ys_test)


def plot():
    
    def norm_img(x):
        return ((x - np.min(x))) / (np.max(x) - np.min(x))
    
    rand_img = np.random.randint(0, 19, 1)[0]

    # take image from target domain and transform with generator 
    img = G(Variable(torch.from_numpy(Xt_train[:20]))).data.numpy()[rand_img]
    
    # plot generated, target, and source images
    subplot(1,3,1)
    plt.imshow(norm_img(img.transpose(1,2,0)).squeeze())
    subplot(1,3,2)
    plt.imshow(norm_img(Xt_train[rand_img].transpose(1,2,0)).squeeze())
    subplot(1,3,3)
    plt.imshow(norm_img(Xs_train[rand_img].transpose(1,2,0)).squeeze())
    
    # print plot
    plt.show()
    
    return rand_img



'''
MODELS
'''

# The generator model transforms target data as though it is sampled from source distribution
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, output_size, kernel_size=3, stride=1, padding=1)
        self.drop = nn.Dropout2d(.5)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))                   
        x = self.drop(F.leaky_relu(self.bn(self.conv2(x))))
        x = self.drop(self.bn(self.conv2(x)))
        x = self.conv3(x)
        return F.tanh(x)

# The discriminator model tries to distinguish between source and target data
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn = nn.BatchNorm2d(20)
        self.drop = nn.Dropout2d(.25)
        self.fc1 = nn.Linear(320, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.leaky_relu(F.max_pool2d(self.conv1(x), 2))
        x = F.leaky_relu(F.max_pool2d(self.drop(self.bn(self.conv2(x))), 2))
        x = x.view(-1, 320)
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return F.sigmoid(x)
    
# The classifier model tries to guess the label of both source and target data
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=2)
        self.bn = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(432, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_classes)
        self.drop = nn.Dropout2d(.25)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=1)
        x = F.max_pool2d(self.bn(self.conv2(x)), kernel_size=2, stride=1)
        x = x.view(-1, 432)
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = F.leaky_relu(self.drop(self.fc2(x)))
        return F.softmax(self.fc3(x))


# model/training parameters
print_interval = 1000         # print example images every 'print_interval' iterations
d_hidden_size = 150           # Discriminator complexity
c_hidden_size = 50            # Classifier complexity
d_learning_rate = 5e-5        # learning rate for discriminator
g_learning_rate = 5e-5        # learning rate for generator
c_learning_rate = 1e-3        # learning rate for classifier
l = 0.3                       # regularizer for adversarial loss
num_epochs = 4                # number of epochs to train models
clf_epochs = 3                # number of epochs to pretrain classifier
batch_size = 50               # size of image sample per epoch

# data parameters
source_data = 'cabauw'        # mnist / cabauw
target_data = 'bilt'          # mnist_m / bilt
input_size = 3                # 3 channels for RGB images, 1 for Greyscale
output_size = 3               # idem above
n_class = 4                   # number of classes; mnist experiment = 10, knmi = 4

# approximate labeling parameters
method = 'default'            # default / selective / iterative
n_val = 1000                  # approximate labeling validation set size
update_delay = 200            # iterations per approximate label update


# load source domain dataset
(Xs_train, ys_train), (Xs_test, ys_test) = data_loader(source_data)
source_batches = make_batches(Xs_train.shape[0], batch_size)

# load target domain dataset
(Xt_train, yt_train), (Xt_test, yt_test) = data_loader(target_data)
target_batches = make_batches(Xt_train.shape[0], batch_size)

# set number of splits
n_split = int(round(Xt_train.shape[0] / batch_size))

# instantiate the models
G = Generator(input_size=input_size, output_size=output_size)
D = Discriminator(input_size=output_size, hidden_size=d_hidden_size)
C = Classifier(input_size=output_size, hidden_size=c_hidden_size, n_classes=n_class)

# define loss functions
bce_crit = nn.BCELoss() # binary crossentropy
mse_crit = nn.MSELoss() # mean squared error

# set the optimizers
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=(0.9, 0.999))
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=(0.9, 0.999))
c_optimizer = optim.Adam(C.parameters(), lr=c_learning_rate, betas=(0.9, 0.999))


# Pre-train the classifier on source dataset
print('\n\nPre-training classifier model on {} dataset..'.format(source_data.upper()))
for epoch in range(clf_epochs):
    print('\nEpoch {}/{}'.format(epoch+1, clf_epochs))
    
    # iterate over batches
    generator = batch_gen(source_batches, np.arange(Xs_train.shape[0]), Xs_train, ys_train)
    for batch in range(int(round(Xs_train.shape[0] / batch_size))):
        
        # reset gradients
        C.zero_grad()
        
        # get predictions from batch
        Xs, Ys, _ = next(generator)
        c_out = C(Xs)
        
        # forward and backward pass through classifier
        c_error = mse_crit(c_out, Ys.float())
        c_error.backward()
        c_optimizer.step()
    
    # evaluate on validation set
    Xstest = Variable(torch.from_numpy(Xs_test))
    Ystest = Variable(torch.from_numpy(ys_test))
    c_val_out = C(Xstest)
    c_val_error = mse_crit(c_val_out, Ystest.float())
    
    # print losses
    print(' - train_mse: {} - val_mse: {}'.format(format(c_error.data[0], '.4f'),
                                                format(c_val_error.data[0], '.4f')))

print('\nEvaluating classifier performance on {} and {} dataset..'.format(source_data.upper(), target_data.upper()))

# create approximate labels Ystar
y_star = C(Variable(torch.from_numpy(Xt_train)))
t_classes = y_star.max(1)[1].data.numpy().ravel()

# get subset of source test
Ys = C(Variable(torch.from_numpy(Xs_test[:1000])))
s_classes = Ys.max(1)[1].data.numpy().ravel()

# source data performance
upper_lim = accuracy_score(y_true=[np.argmax(i) for i in ys_test[:1000]], y_pred=s_classes)
print('{} accuracy: {}'.format(source_data.upper(), round(upper_lim, 3)))

# approximate label performance
lower_lim = accuracy_score(y_true=[np.argmax(i) for i in yt_train], y_pred=t_classes)
print('{} accuracy: {}'.format(target_data.upper(), round(lower_lim, 3)))

# obtain approximate labels for target set
y_appr = np_utils.to_categorical(t_classes, n_class)
X_val = Xt_test[:n_val]

# create lists for losses and accuracies
acc_vec = []
d_loss_vec = []
c_loss_vec = []
update_iter = []
best_acc = lower_lim


# train the approximate label matcher model
print('\n\nTraining discriminator and generator..')
for epoch in range(num_epochs):
    print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
    start_time = time.time()
    
    # prepare source and target batch
    source_gen = batch_gen(source_batches, np.arange(Xs_train.shape[0]), Xs_train, ys_train)
    target_gen = batch_gen(target_batches, np.arange(Xt_train.shape[0]), Xt_train, None)
    
    # iterate over batches
    for i, batch in enumerate(range(int(round(Xs_train.shape[0] / batch_size)))):
    
        # get batch of source, target data and discriminator target
        Xs, _, split_idx = next(source_gen)
        Xt, split_idx = next(target_gen)        
        y_ones = Variable(torch.ones((len(Xt), 1)))     # ones for source
        y_zeros = Variable(torch.zeros((len(Xt), 1)))   # zeros for target
        
        # failsafe for batch generator
        if len(Xs) != batch_size:
            continue
        if len(Xt) != batch_size:
            continue

        # get approx labels for batch
        y_approx = Variable(torch.from_numpy(y_appr[split_idx]))

        # 1) TRAIN THE DISCRIMINATOR
        D.zero_grad()

        # 1a) Train the discriminator on source images
        d_out = D(Xs)
        d_source_loss = bce_crit(d_out, y_ones) * l
        d_source_loss.backward(retain_variables=True)


        # 1b) Train the discriminator on target images
        d_out = D(G(Xt))
        d_target_loss = bce_crit(d_out, y_zeros) * l
        d_target_loss.backward(retain_variables=True)
        d_optimizer.step()

        # 2) TRAIN THE GENERATOR
        G.zero_grad()

        # 2a) Train the generator on discriminator response
        d_out = D(G(Xt))
        g_d_loss = bce_crit(d_out, y_ones) * l   # fool discriminator (ones for target)
        g_d_loss.backward(retain_variables=True)
        g_optimizer.step()

        # 2b) Train the generator on classifier response
        G.zero_grad()

        c_out = C(G(Xt))
        g_c_loss = mse_crit(c_out, y_approx.float()) * (1 - l)
        g_c_loss.backward(retain_variables=True)
        g_optimizer.step()


        # 3) Print losses and target validation accuracy
        t_classes = C(G(Variable(torch.from_numpy(X_val)))).max(1)[1].data.numpy().ravel()
        acc = accuracy_score(y_true=[np.argmax(i) for i in yt_test[:n_val]], y_pred=t_classes)
        
        if (i+1) % print_interval == 0:
            print(' - {}s - d_{}: {} - d_{}: {} - g-d: {} - g-c: {} - {}_val_acc: {}\n'.format(
                        round(time.time() - start_time),
                        source_data, format(d_source_loss.data[0], '.4f'), 
                        target_data, format(d_target_loss.data[0], '.4f'), 
                        format(g_d_loss.data[0], '.4f'),
                        format(g_c_loss.data[0], '.4f'),
                        target_data, format(acc, '.4f')), end='')
        
        
        # save losses to vector
        acc_vec.append(float(format(acc, '.4f')))
        d_loss_vec.append(float(format(g_d_loss.data[0], '.4f')))
        c_loss_vec.append(float(format(g_c_loss.data[0], '.4f')))

        # selective label update method
        if method == 'selective':
            # remember best acc with labels
            new_best = np.max([np.max(acc_vec), lower_lim])
            if new_best > best_acc:
                print('\nUpdating approximate labels. Best label accuracy: {}'.format(format(new_best, '.4f')))

                # update approximate labels
                y_star = []
                n_splits = int(round(Xt_train.shape[0] / batch_size))
                gen = batch_generator(Xt_train, yt_train, n_splits)
                for xt in gen:
                    pred = C(G(xt[0]))
                    y_star.append(pred.max(1)[1].data.numpy().ravel())
                y_appr = np_utils.to_categorical([x for y in y_star for x in y], n_class)

                # remember new best accuracy and save iteration of update
                best_acc = new_best
                update_iter.append(int((epoch*i)+i))
                
        # iterative label update method
        if method == 'iterative':
            if (epoch * n_split + (i+1)) % update_delay == 0:
                print('Updating approximate labels..')
                y_star = []
                n_splits = int(round(Xt_train.shape[0] / batch_size))
                gen = batch_generator(Xt_train, yt_train, n_splits)
                for xt in gen:
                    pred = C(G(xt[0]))
                    y_star.append(pred.max(1)[1].data.numpy().ravel())
                y_appr = np_utils.to_categorical([x for y in y_star for x in y], n_class)

        # plot example images
        if (i+1) % print_interval == 0:
            img = plot()
            print('Approximate label: {}, True label: {}.\n'.format(np.argmax(y_appr[img]), np.argmax(yt_train[img])))

  
# create predictions with fake target images
y_star = []
n_split = int(round(Xt_test.shape[0] / batch_size))
target_gen = batch_generator(Xt_test, yt_test, n_split)
for xt in target_gen:
    pred = C(G(xt[0]))
    y_star.append(pred.max(1)[1].data.numpy().ravel())

# return accuracy and confusion matrix 
acc = accuracy_score(y_true=[np.argmax(i) for i in yt_test], y_pred=[x for y in y_star for x in y])
print('\n{} test accuracy: {}'.format(target_data.upper(), round(acc, 3)))
