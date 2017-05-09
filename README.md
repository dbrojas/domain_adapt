# Iterative Learning in Domain Adaptation Networks

## Models
dann_mnist.ipynb -- DANN (Ganin, 2016) on MNIST and MNIST-M (Keras).

alm_mnist_usps.py -- ALM (Ash, 2017) on MNIST and USPS (PyTorch).

fog_detection.ipynb -- Fog detection with InceptionV3 on De Bilt and Cabauw weather station images (Keras).

## Keras modifications
keras/*

DANN model makes use of the Gradient Reversal layer. This folder contains files which need to be replaced with original Keras files to use this layer in the module.
