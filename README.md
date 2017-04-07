# Iterative Learning in Domain Adaptation Networks

## Models
dann_mnist.ipynb -- DANN (Ganin, 2016) on MNIST-M.

fog_detection.ipynb -- VGG16 and InceptionV3 stacked for fog detection.

## Keras modifications
keras/*

Usually, domain adaptation networks make use of the Gradient Reversal layer. This folder contains files which need to be replaced with original Keras files to use this layer in the module.
