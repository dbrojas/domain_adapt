# Fog Detection with Domain Adaptation Networks

Fog is a difficult to predict and very local weather phenomena. With computer vision, development and movement of fog can be monitored with camera systems. Image classification algorithms can help classify the density of the fog that is present on the captured image. Images accompanied with a MOR visibility label (obtained from scatterometers) can be used to train these algorithms.

The problem is that fog must also be detectable from locations other than where scatterometers are available. This highlights the importance of image classification algorithms to be sensitive to fog in other locations as well. In other words, the algorithm should generalize.

In this repository, several deep convolutional neural networks are evaluated for their generalization capacity to recognize fog in other locations. Also, some experiments with domain adaptation networks (the field of study concerned with generalization) can be found. The DANN (Ganin et al., 2016) and ALM (Ash et al., 2016) have been implemented.

A good place to start experimenting with the domain adaptation networks is to use the alm_mnistm_eval.ipynb script. Here, ALM is used to align MNIST-M images to MNIST distributions.
