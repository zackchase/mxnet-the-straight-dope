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

## Table of Contents 
* [Manipulating data with NDArray](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/1-ndarray.ipynb) 
* [Automatic differentiation via ``autograd``](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/2-autograd.ipynb)
* [Linear Regression *(from scratch!)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/3-linear-regression-scratch.ipynb)
* [Linear Regression *(with gluon!)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/4-linear-regression-gluon.ipynb)
* [Multiclass Logistic Regression *(from scratch!)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/5-softmax-regression-scratch.ipynb)
