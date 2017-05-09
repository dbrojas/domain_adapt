#!/usr/bin/env python
# coding: utf-8

# Approximate Label Matching (ALM; Ash et al., 2017) implementation in PyTorch.
# Read the paper at https://arxiv.org/pdf/1602.04889.pdf
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from keras.datasets import mnist
from keras.utils import np_utils


# Model parameters
input_size = 1                # n_filters for target set; USPS has 1 greyscale filter
output_size = 1               # Transform to source set; MNIST has 1 greyscale filter
d_hidden_size = 50            # Discriminator complexity
c_hidden_size = 50            # Classifier complexity
d_learning_rate = 4e-4        # learning rate for discriminator
g_learning_rate = 4e-4        # learning rate for generator
c_learning_rate = 1e-3        # learning rate for classifier
l = 0.2                       # regularizer for loss
optim_betas = (0.9, 0.999)    # optimizer betas
num_epochs = 15               # number of epochs to train models
batch_size = 32               # size of image sample per epoch 
print_interval = 100          # interval to print losses in epochs


# ##### DATA
# load source domain dataset
(Xs_train, ys_train), (Xs_test, ys_test) = mnist.load_data()
Xs_train = torch.from_numpy(Xs_train.reshape(Xs_train.shape[0], 1, 28, 28).astype('float32') / 255)
Xs_test = torch.from_numpy(Xs_test.reshape(Xs_test.shape[0], 1, 28, 28).astype('float32') / 255)
ys_train = torch.from_numpy(np_utils.to_categorical(ys_train, 10))
ys_test = torch.from_numpy(np_utils.to_categorical(ys_test, 10))

# load target domain dataset
def normalize_img(x):
    for channel in range(x.shape[0]):
        for width in range(x.shape[1]):
            for height in range(x.shape[2]):
                x[channel, width, height] = (x[channel, width, height] - np.min(x)) / (np.max(x) - np.min(x))
    return x

with open('../data/usps/zip.train') as f:
    Xt_train = f.readlines()
with open('../data/usps/zip.test') as f:
    Xt_test = f.readlines()
    
# one-hot encode target, preprocess raw images 
yt_train = torch.from_numpy(np_utils.to_categorical(np.array([int(digit[0]) for digit in Xt_train]), 10))
yt_test = torch.from_numpy(np_utils.to_categorical(np.array([int(digit[0]) for digit in Xt_test]), 10))
Xt_train = np.array([np.delete(np.array(p.strip().split(' ')).astype('float32'), 0) for p in Xt_train]).reshape(len(yt_train), 1, 16, 16)
Xt_test = np.array([np.delete(np.array(p.strip().split(' ')).astype('float32'), 0) for p in Xt_test]).reshape(len(yt_test), 1, 16, 16)

# zero-pad usps images to shape (1 x 28 x 28)
Xt_train = np.array([np.pad(img, ((0,0),(6,6),(6,6)), 'constant', constant_values=-1) for img in Xt_train])
Xt_test = np.array([np.pad(img, ((0,0),(6,6),(6,6)), 'constant', constant_values=-1) for img in Xt_test])

# normalize image values between 0 and 1
Xt_train = torch.from_numpy(np.array([normalize_img(img) for img in Xt_train]))
Xt_test = torch.from_numpy(np.array([normalize_img(img) for img in Xt_test]))


# ##### MODELS
# The generator model transforms target data as though it is sampled from source distribution
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, output_size, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.conv1(x))                   
        y = F.relu(self.bn(self.conv2(x)))
        y = self.bn(self.conv2(y))
        x = x + y
        x = self.conv3(x)
        return F.tanh(x)

# The discriminator model tries to distinguish between source and target data
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.sigmoid(x)
    
# The classifier model tries to guess the label of both source and target data
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(432, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=1)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=1)
        x = x.view(-1, 432)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return F.softmax(self.fc3(x))
    
# Instantiate the models
G = Generator(input_size=input_size, output_size=output_size)
D = Discriminator(input_size=output_size, hidden_size=d_hidden_size)
C = Classifier(input_size=output_size, hidden_size=c_hidden_size)

# Define loss functions
bce_crit = nn.BCELoss() # binary crossentropy
mse_crit = nn.MSELoss() # mean squared error: used to minimize equation 1.

# Set the optimizers
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)
c_optimizer = optim.Adam(C.parameters(), lr=c_learning_rate, betas=optim_betas)




# Pre-train the classifier on source dataset
clf_epochs = 1500
print('\nPre-training classifier model on MNIST dataset..')
for epoch in range(1, clf_epochs):      
    C.zero_grad()

    Xs = Variable(Xs_train[(batch_size*epoch):(batch_size*(epoch+1))])
    Ys = Variable(ys_train[(batch_size*epoch):(batch_size*(epoch+1))])

    c_out = C(Xs)
    c_error = mse_crit(c_out, Ys.float())
    c_error.backward()

    c_optimizer.step()

    if (epoch+1) % print_interval == 0:
        print('Epoch {}/{} - mse: {}'.format(epoch+1, clf_epochs, format(c_error.data[0], '.4f')))




print('\nEvaluating classifier performance on MNIST and USPS dataset..')
# create approximate labels Ystar
Ystar = C(Variable(Xt_train))
_, t_classes = Ystar.max(1)

# get subset of source test
Ys = C(Variable(Xs_test[:1000]))
_, s_classes = Ys.max(1)

# source data performance
acc = accuracy_score(y_true=[np.argmax(i)+1 for i in ys_test[:1000].numpy()], y_pred=s_classes.data.numpy().ravel()+1)
print('MNIST accuracy: {}'.format(round(acc, 3)))

# approximate label performance
acc = accuracy_score(y_true=[np.argmax(i)+1 for i in yt_train.numpy()], y_pred=t_classes.data.numpy().ravel()+1)
print('USPS accuracy: {}'.format(round(acc, 3)))



# obtain approximate labels for target set
Ytrain = torch.from_numpy(np_utils.to_categorical(t_classes.data.numpy(), 10))
X_val = Variable(Xt_test[:100])

# initialise errors
err_real = 0
err_fake = 0
gen_err = 0
err = 0

# Train the models
print('\nTraining discriminator and generator..')
for epoch in range(num_epochs):
    
    # 1) TRAIN THE DISCRIMINATOR
    D.zero_grad()

    # 1a) Train the discriminator on source images
    Xs = Variable(Xs_train[(batch_size*epoch):(batch_size*(epoch+1))])    # batch from source dataset
    Yd = Variable(torch.ones((batch_size, 1)))                            # ones for source dataset
    d_out = D(Xs)                                                         # forward pass through discriminator
    err_real = bce_crit(d_out, Yd) * l                                    # calculate binary crossentropy
    err_real.backward(retain_variables=True)                              # compute/store gradients, but don't change params yet
    d_optimizer.step()                                                    # optimise discriminator parameters


    # 1b) Train the discriminator on target images
    Xt = Variable(Xt_train[(batch_size*epoch):(batch_size*(epoch+1))])    # batch from target dataset
    Yd = Variable(torch.zeros((batch_size, 1)))                           # zeros for target dataset
    d_out = D(G(Xt))                                                      # forward pass through discriminator
    err_fake = bce_crit(d_out, Yd) * l                                    # calculate binary crossentropy
    err_fake.backward(retain_variables=True)                              # compute gradients
    d_optimizer.step()                                                    # optimise discriminator parameters
    
    
    # 2) TRAIN THE GENERATOR
    G.zero_grad()

    # 2a) Train the generator on discriminator response (but DO NOT train D on these labels)
    Yd = Variable(torch.ones((batch_size, 1)))                            # ones for target (we want to fool now) 
    d_out = D(G(Xt))                                                      # forward pass through discriminator
    gen_err = bce_crit(d_out, Yd) * l                                     # calculate binary crossentropy with fooled labels
    gen_err.backward(retain_variables=True)                               # compute gradients
    g_optimizer.step()                                                    # optimise generator parameters

    # 2b) Train the generator on classifier response (but DO NOT train C on these labels)
    G.zero_grad()

    Ystar = Variable(Ytrain[(batch_size*epoch):(batch_size*(epoch+1))])   # approximate labels for target data
    c_out = C(G(Xt))                                                      # forward pass through discriminator
    err = mse_crit(c_out, Ystar.float()) * (1 - l)                        # calculate mean squared error with class labels
    err.backward(retain_variables=True)                                   # compute gradients
    g_optimizer.step()                                                    # optimise generator parameters

    
    # 3) PRINT ERROR RATES
    if epoch % 1 == 0:
        Y_val = C(G(X_val))
        _, t_classes = Y_val.max(1)
        acc = accuracy_score(y_true=[np.argmax(i)+1 for i in yt_test[:100].numpy()], y_pred=t_classes.data.numpy().ravel()+1)
        print('Epoch {}/{} -- d_mnist_loss: {} - d_usps_loss: {} - g-d_loss: {} - g-c_loss: {} - usps_acc: {}'.format(
                      epoch+1, num_epochs, 
                      format(err_real.data[0], '.4f'), 
                      format(err_fake.data[0], '.4f'), 
                      format(gen_err.data[0], '.4f'),
                      format(err.data[0], '.4f'),
                      format(acc, '.4f')))









# ''' This block contains useful snippets of code when
# MNIST-M is used with an Approximate Label Matcher '''

# # The classifier model tries to guess the label of both source and target data
# class Approx_label(nn.Module):
#     def __init__(self, input_size, hidden_size, input_shape=(3, 28, 28)):
#         super(Approx_label, self).__init__()
#         self.conv1 = nn.Conv2d(input_size, 32, kernel_size=5, stride=2)
#         self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=2)
#         self.fc1 = nn.Linear(432, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc3 = nn.Linear(hidden_size, 10)

#     def forward(self, x):
#         x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=1)
#         x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=1)
#         x = x.view(-1, 432)
#         x = F.elu(self.fc1(x))
#         x = F.elu(self.fc2(x))
#         return F.softmax(self.fc3(x))

# A = Approx_label(input_size=input_size, hidden_size=c_hidden_size)
# a_optimizer = optim.Adam(A.parameters(), lr=c_learning_rate, betas=optim_betas)

# ##### MNIST-M data
# mnistm = mnist_m.load_data()
# Xt_train = torch.from_numpy(mnistm[b'train'].reshape(60000, 3, 28, 28).astype('float32') / 255)
# Xt_test = torch.from_numpy(mnistm[b'test'].reshape(10000, 3, 28, 28).astype('float32') / 255)
# yt_train, yt_test = ys_train, ys_test

# # Create dataset for approximate labelling
# (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
# xtrain = torch.from_numpy(np.array([np.array(Image.fromarray(img).convert('RGB')).reshape(3,28,28).astype('float32')/255 for img in xtrain]))
# xtest = torch.from_numpy(np.array([np.array(Image.fromarray(img).convert('RGB')).reshape(3,28,28).astype('float32')/255 for img in xtest]))

