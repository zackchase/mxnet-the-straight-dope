# Deep Learning - The Straight Dope

## Abstract
This repo contains an
incremental sequence of notebooks designed to teach deep learning, MXNet, and
the ``gluon`` interface. Our goal is to leverage the strengths of Jupyter
notebooks to present prose, graphics, equations, and code together in one place.
If we're successful, the result will be a resource that could be simultaneously
a book, course material, a prop for live tutorials, and a resource for
plagiarising (with our blessing) useful code. To our knowledge there's no source
out there that teaches either (1) the full breadth of concepts in modern deep
learning or (2) interleaves an engaging textbook with runnable code. We'll find
out by the end of this venture whether or not that void exists for a good
reason.

Another unique aspect of this book is its authorship process. We are
developing this resource fully in the public view and are making it available
for free in its entirety. While the book has a few primary authors to set the
tone and shape the content, we welcome contributions from the community and hope
to coauthor chapters and entire sections with experts and community members.
Already we've received contributions spanning typo corrections through full
working examples.  

## Implementation with Apache MXNet
Throughout this book,
we rely upon MXNet to teach core concepts, advanced topics, and a full
complement of applications. MXNet is widely used in production environments
owing to its strong reputation for speed. Now with ``gluon``, MXNet's new
imperative interface (alpha), doing research in MXNet is easy. 

## Dependencies
To run these notebooks, you'll want to build MXNet from source. Fortunately,
this is easy (especially on Linux) if you follow [these
instructions](http://mxnet.io/get_started/install.html). You'll also want to
[install Jupyter](http://jupyter.readthedocs.io/en/latest/install.html) and use
Python 3 (because it's 2017). 

## Slides

The authors (& others) are
increasingly giving talks that are based on the content in this books. Some of
these slide-decks (like the 6-hour KDD 2017) are gigantic so we're collecting
them separately in [this repo](https://github.com/zackchase/mxnet-slides).
Contribute there if you'd like to share tutorials or course material based on
this books.

## Table of contents 

### Part 1: Deep Learning Fundamentals 
* **Chapter 1:** Crashcourse
    * [Preface](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter01_crashcourse/preface.ipynb)
    * [Introduction](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter01_crashcourse/introduction.ipynb)
    * [Manipulating data with NDArray](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter01_crashcourse/ndarray.ipynb)
    * [Linear algebra](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter01_crashcourse/linear-algebra.ipynb)
    * [Probability and statistics](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter01_crashcourse/probability.ipynb)
    * [Automatic differentiation via ``autograd``](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter01_crashcourse/autograd.ipynb)

* **Chapter 2:** Introduction to Supervised Learning
    * [Linear regression *(from scratch)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter02_supervised-learning/linear-regression-scratch.ipynb)
    * [Linear regression *(with ``gluon``)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter02_supervised-learning/linear-regression-gluon.ipynb)
    * [Perceptron and SGD primer](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter02_supervised-learning/perceptron.ipynb)
    * [Multiclass logistic regression *(from scratch)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter02_supervised-learning/softmax-regression-scratch.ipynb)
    * [Multiclass logistic regression *(with ``gluon``)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter02_supervised-learning/softmax-regression-gluon.ipynb)
    * [Overfitting and regularization *(from scratch)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter02_supervised-learning/regularization-scratch.ipynb)
     * [Overfitting and regularization *(with ``gluon``)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter02_supervised-learning/regularization-gluon.ipynb)
     * [Loss functions](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter02_supervised-learning/loss.ipynb)
     * [Learning environments](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter02_supervised-learning/environment.ipynb)

* **Chapter 3:** Deep neural networks (DNNs) 
    * [Multilayer perceptrons *(from scratch)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter03_deep-neural-networks/mlp-scratch.ipynb)
    * [Multilayer perceptrons *(with ``gluon``)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter03_deep-neural-networks/mlp-gluon.ipynb)
    * [Dropout regularization (from scratch)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter03_deep-neural-networks/mlp-dropout-scratch.ipynb)
    * [Dropout regularization (with ``gluon``)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter03_deep-neural-networks/mlp-dropout-gluon.ipynb)
    * [Introduction to ``gluon.Block`` and ``gluon.nn.Sequential()``](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter03_deep-neural-networks/plumbing.ipynb)
    * [Writing custom layers with ``gluon.Block``](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter03_deep-neural-networks/custom-layer.ipynb)
    * [Serialization: saving and loading models](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter03_deep-neural-networks/serialization.ipynb)
    * Advanced Data IO
    * Debugging your neural networks

* **Chapter 4:** Convolutional neural networks (CNNs) 
     * [Convolutional neural networks *(from scratch!)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter04_convolutional-neural-networks/cnn-scratch.ipynb)
     * [Convolutional neural networks *(with ``gluon``!)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter04_convolutional-neural-networks/cnn-gluon.ipynb)
     * [Introduction to deep CNNs (AlexNet)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter04_convolutional-neural-networks/deep-cnns-alexnet.ipynb)
     * [Very deep networks and repeating blocks (VGG network)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter04_convolutional-neural-networks/very-deep-nets-vgg.ipynb)
     * [Batch normalization (from scratch)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter04_convolutional-neural-networks/cnn-batch-norm-scratch.ipynb)
     * [Batch normalization (with ``gluon``)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter04_convolutional-neural-networks/cnn-batch-norm-gluon.ipynb)

* **Chapter 5:** Recurrent neural networks (RNNs)
    * [Simple RNNs (from scratch)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter05_recurrent-neural-networks/simple-rnn.ipynb)
    * [LSTMS RNNs (from scratch)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter05_recurrent-neural-networks/lstm-scratch.ipynb)
    * [GRUs (from scratch)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter05_recurrent-neural-networks/gru-scratch.ipynb)
    * [RNNs (with ``gluon``)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter05_recurrent-neural-networks/rnns-gluon.ipynb)
    * ***Roadmap*** Dropout for recurrent nets
    * ***Roadmap*** Zoneout regularization


* **Chapter 6:** Optimization
    * [Introduction to optimization](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter06_optimization/optimization-intro.ipynb)
    * [Gradient descent and stochastic gradient descent](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter06_optimization/gd-sgd.ipynb)
    * ***Roadmap*** Momentum
    * ***Roadmap*** AdaGrad
    * ***Roadmap*** RMSProp
    * ***Roadmap*** Adam 
    * ***Roadmap*** AdaDelta
    * ***Roadmap*** SGLD / SGHNT

* **Chapter 7:** Distributed & high-performance learning
    * [Fast & flexible: combining imperative & symbolic nets with HybridBlocks](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter07_distributed-learning/hybridize.ipynb)
    * [Training with multiple GPUs (from scratch)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter07_distributed-learning/multiple-gpus-scratch.ipynb) 
    * [Training with multiple GPUs (with ``gluon``)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter07_distributed-learning/multiple-gpus-gluon.ipynb)
    * [Training with multiple machines](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter07_distributed-learning/training-with-multi-machines.ipynb)
    * ***Roadmap*** Distributed optimization (Asynchronous SGD, ...)
    * ***Roadmap*** Combining imperative deep learning with symbolic graphs

### Part 2: Applications
* **Chapter 8:** Computer vision (CV) 
    * ***Roadmap*** Network of networks (inception & co)
    * ***Roadmap*** Residual networks
    * [Object detection](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter08_computer-vision/object-detection.ipynb)
    * ***Roadmap*** Fully-convolutional networks
    * ***Roadmap*** Siamese (conjoined?) networks
    * ***Roadmap*** Embeddings (pairwise and triplet losses)
    * ***Roadmap*** Inceptionism / visualizing feature detectors
    * ***Roadmap*** Style transfer
    * [Fine-tuning](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter08_computer-vision/fine-tuning.ipynb)

* **Chapter 9:** Natural language processing (NLP)
    * ***Roadmap*** Word embeddings (Word2Vec)
    * ***Roadmap*** Sentence embeddings (SkipThought)
    * ***Roadmap*** Sentiment analysis
    * ***Roadmap*** Sequence-to-sequence learning (machine translation)
    * ***Roadmap*** Sequence transduction with attention (machine translation)
    * ***Roadmap*** Named entity recognition 
    * ***Roadmap*** Image captioning
    * [Tree-LSTM for semantic relatedness](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter09_natural-language-processing/tree-lstm.ipynb)
 
* **Chapter 10:** Audio Processing
    * ***Roadmap*** Intro to automatic speech recognition
    * ***Roadmap*** Connectionist temporal classification (CSC) for unaligned sequences
    * ***Roadmap*** Combining static and sequential data

* **Chapter 11:** Recommender systems
    * ***Roadmap*** Latent factor models
    * ***Roadmap*** Deep latent factor models
    * ***Roadmap*** Bilinear models
    * ***Roadmap*** Learning from implicit feedback

* **Chapter 12:** Time series
    * ***Roadmap*** Forecasting
    * ***Roadmap*** Modeling missing data
    * ***Roadmap*** Combining static and sequential data

### Part 3: Advanced Methods
* **Chapter 13:** Unsupervised learning
   * ***Roadmap*** Introduction to autoencoders
   * ***Roadmap*** Convolutional autoencoders (introduce upconvolution)
   * ***Roadmap*** Denoising autoencoders
   * ***Roadmap*** Variational autoencoders
   * ***Roadmap*** Clustering
   
* **Chapter 14:** Generative adversarial networks (GANs)
    *[Introduction to GANs](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter14_generative-adversarial-networks/gan-intro.ipynb)
    * [Deep convolutional GANs (DCGANs)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter14_generative-adversarial-networks/dcgan.ipynb)
    * ***Roadmap*** Wasserstein-GANs
    * ***Roadmap*** Energy-based GANS
    * ***Roadmap*** Conditional GANs
    * ***Roadmap*** Image transduction GANs (Pix2Pix)
    * ***Roadmap*** Learning from Synthetic and Unsupervised Images 

* **Chapter 15:** Adversarial learning
    * ***Roadmap*** Two Sample Tests
    * ***Roadmap*** Finding adversarial examples
    * ***Roadmap*** Adversarial training

* **Chapter 16:** Tensor Methods
    * ***Roadmap*** Introduction to tensor algebra
    * ***Roadmap*** Tensor decomposition
    * ***Roadmap*** Tensorized neural networks

* **Chapter 17:** Deep reinforcement learning (DRL)
    * ***Roadmap*** Introduction to reinforcement learning
    * ***Roadmap*** Deep contextual bandits
    * [Deep Q-networks](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter17_Deep-Reinforcement-Learning/DQN.ipynb)
    * ***Roadmap*** Policy gradient
    * ***Roadmap*** Actor-critic gradient

* **Chapter 18:** Variational methods and uncertainty
    * ***Roadmap*** Dropout-based uncertainty estimation (BALD)
    * ***Roadmap*** Weight uncertainty (Bayes-by-backprop)
    * ***Roadmap*** Variational autoencoders

### Appendices
* Appendix 1: Cheatsheets
    * ***Roadmap*** ``gluon`` 
    * ***Roadmap*** PyTorch to MXNet
    * ***Roadmap*** Tensorflow to MXNet
    * ***Roadmap*** Keras to MXNet
    * ***Roadmap*** Math to MXNet


## Choose your own adventure
We've designed these tutorials so that you can traverse the curriculum in more than one way.
* Anarchist - Choose whatever you want to read, whenever you want to read it.
* Imperialist - Proceed through all tutorials in order. In this fashion you will be exposed to each model first from scratch, writing all the code ourselves but for the basic linear algebra primitives and automatic differentiation.
* Capitalist - If you don't care how things work (or already know) and just want to see working code in ``gluon``, you can skip (*from scratch!*) tutorials and go straight to the production-like code using the high-level ``gluon`` front end.

## Authors
This evolving creature is a collaborative effort (see contributors tab). The lead writers, assimilators, and coders include:
* Zachary C. Lipton ([@zackchase](https://github.com/zackchase))
* Mu Li ([@mli](https://github.com/mli))
* Alex Smola ([@smolix](https://github.com/smolix))
* Sheng Zha ([@szha](https://github.com/szha))
* Aston Zhang ([@astonzhang](https://github.com/astonzhang))
* Joshua Z. Zhang ([@zhreshold](https://github.com/zhreshold))
* Eric Junyuan Xie ([@piiswrong](https://github.com/piiswrong))


## Inspiration 
In creating these tutorials, we've have drawn inspiration from some the resources that allowed us
to learn deep / machine learning with other libraries in the past. These include:

* [Soumith Chintala's *Deep Learning with PyTorch: A 60 Minute Blitz*](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
* [Alec Radford's *Bare-bones intro to Theano*](https://github.com/Newmu/Theano-Tutorials) 
* [Video of Alec's intro to deep learning with Theano](https://www.youtube.com/watch?v=S75EdAcXHKk)
* [Chris Bishop's *Pattern Recognition and Machine Learning*](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738)

## Contribute
* Already, in the short time this project has been off the ground, we've gotten some helpful PRs from the community with pedagogical suggestions, typo corrections, and other useful fixes. If you're inclined, please contribute!
