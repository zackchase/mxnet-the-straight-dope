# mxnet-the-straight-dope
Much easy, so MXNet. Wow.

## Abstract
MXNet is widely used in production environments owing to its strong reputation for speed. Now with ``gluon``, MXNet's new imperative interface (alpha), doing research in MXNet is easy. 

This repo contains an incremental sequence of tutorials to make learning ``gluon`` easy (meta-easy?). These tutorials are designed with public presentations in mind and are composed as Jupyter notebooks where each notebook corresponds to roughly 20 minutes of rambling and each codeblock could correspond to roughly one slide.


## Inspiration 

In creating these tutorials, I've borrowed heavily from some of the resources that were most useful when I learned how to program with Theano and PyTorch, specifically:

* [Soumith Chintala's helladope "Deep Learning with PyTorch: A 60 Minute Blitz"](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
* [Alec Radford's bare-bones intro to Theano](https://github.com/Newmu/Theano-Tutorials) 
* [Video of Alec's awesome intro to deep learning with Theano](https://www.youtube.com/watch?v=S75EdAcXHKk)

## Preliminaries

To run these notebooks, you'll want to build mxnet from source. Fortunately, this is easy (especially on Linux) if you follow [these instructions](http://mxnet.io/get_started/install.html). You'll also want to [install Jupyter](http://jupyter.readthedocs.io/en/latest/install.html) and use Python 3 (because it's 2017). 

## Table of contents 

### Basics
* [1 - Manipulating data with NDArray](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/1-ndarray.ipynb) 
* [2 - Automatic differentiation via ``autograd``](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/2-autograd.ipynb)
* [3a - Linear Regression *(from scratch!)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/3a-linear-regression-scratch.ipynb)
* [3b - Linear Regression *(with ``gluon``!)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/3b-linear-regression-gluon.ipynb)
* [4a - Multiclass Logistic Regression *(from scratch!)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/4a-softmax-regression-scratch.ipynb)
* [4b - Multiclass Logistic Regression *(with ``gluon``!)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/4b-softmax-regression-gluon.ipynb)
### Intermediate 
* [5a - Multilayer Perceptrons *(from scratch!)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/4a-mlp-scratch.ipynb)
* [5b - Multilayer Perceptrons *(with ``gluon``!)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/4b-mlp-gluon.ipynb)
* [6a - Convolutional Neural Network *(from scratch!)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/6a-cnn-scratch.ipynb)
* [6b - Convolutional Neural Network *(with ``gluon``!)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/6a-cnn-gluon.ipynb)

### Recurrent Neural Networks
* [8a - Simple RNNs and their application to language modeling]()
* **Roadmap** LSTMS
* **Roadmap** GRUs
* **Roadmap** Sequence-to-sequence learning
* **Roadmap** Sequnce transduction with attention

### Computer Vision
* **Roadmap** Residual networks
* **Roadmap** Object detection 
* **Roadmap** Fully-convolutional networks

### Adversarial Learning
* **Roadmap** Finding adversarial examples
* **Roadmap** Adversarial training


### Generative Adversarial Networks
* **Roadmap** Introduction to GANs
* **Roadmap** DCGAN
* **Roadmap** Wasserstein-GANs
* **Roadmap** Energy-based GANS
* **Roadmap** Pix2Pix


### Deep Reinforcement Learning
* **Roadmap** Deep Q-networks
* **Roadmap** Policy gradient
* **Roadmap** Actor-critic gradient

### Distributed learning with *MXNet*
* Multiple GPUs 
* Multiple Machines

### Optimizing MXNet code for production

## Choose your own adventure

I've designed these tutorials so that you can traverse the curriculum in one of three ways.
* Anarchist - Choose whatever you want to read, whenever you want to read it.
* Imperialist - Proceed throught the tutorials in order (1, 2, 3a, 3b, 4a, 4b, ...). In this fashion you will be exposed to each model first from scratch, writing all the code ourselves but for the basic linear algebra primitives and automatic differentiation.
* Capitalist - If you would like to specialize to either the raw interface or the high-level ``gluon`` front end choose either (1, 2, 3a, 4a, ...) or (1, 2, 3b, 4b, ...) respectively.

## Roadmap
* GANs (DCGAN, InfoGAN, EBGAN, Wasserstein GAN, SD-GAN)
* Simple RNN (from scratch and w ``gluon``)
* LSTM (from scratch and w ``gluon``)
* GRU
* DQN 
* Sequence to Sequence 
* Sequence to Sequence with Attention
* Weight uncertainty Bayes-by-Backprop neural networks 
* Residual networks
* Latent factor models
* Word2Vec 

## Collaborators
This evolving creature is a collaborative effort. Some amount of credit (and blame) can be shared by:
* Zachary C. Lipton ([@zackchase](https://github.com/zackchase))
* Mu Li ([@mli](https://github.com/mli))
* Alex Smola ([@smolix](https://github.com/smolix))
* Eric Junyuan Xie ([@piiswrong](https://github.com/piiswrong))
