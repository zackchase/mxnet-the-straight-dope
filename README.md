# Deep Learning - The Straight Dope

## Abstract
This repo contains an incremental sequence of notebooks designed to teach deep learning, MXNet, and the ``gluon`` interface. Our goal is to leverage the strengths of Jupyter notebooks to present prose, graphics, equations, and code together in one place. If we're successful, the result will be a resource that could be simultaneously a book, course material, a prop for live tutorials, and a resource for plagiarising (with our blessing) useful code. To our knowledge there's no source out there that teaches either (1) the full breadth of concepts in modern deep learning or (2) interleaves an engaging textbook with runnable code. We'll find out by the end of this venture whether or not that void exists for a good reason.

Another unique aspect of this book is its authorship process. We are developing this resource fully in the public view and are making it available for free in its entirety. While the book has a few primary authors to set the tone and shape the content, we welcome contributions from the community and hope to coauthor chapters and entire sections with experts and community members. Already we've received contributions spanning typo corrections through full working examples.  

## Implementation with Apache MXNet
Throughout this book, we rely upon MXNet to teach core concepts, advanced topics, and a full complement of applications. MXNet is widely used in production environments owing to its strong reputation for speed. Now with ``gluon``, MXNet's new imperative interface (alpha), doing research in MXNet is easy. 

## Dependencies

To run these notebooks, you'll want to build MXNet from source. Fortunately, this is easy (especially on Linux) if you follow [these instructions](http://mxnet.io/get_started/install.html). You'll also want to [install Jupyter](http://jupyter.readthedocs.io/en/latest/install.html) and use Python 3 (because it's 2017). 

## Table of contents 

### Part 1: Crashcourse 
* [0 - Preface](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P01-C00-preface.ipynb)
* [1 - Introduction](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P01-C01-introduction.ipynb)
* [2 - Manipulating data with NDArray](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P01-C02-ndarray.ipynb) 
* [3 - Linear Algebra](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P01-C03-linear-algebra.ipynb)
* [4 - Probability and Statistics](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P01-C04-probability.ipynb)
* [5 - Automatic differentiation via ``autograd``](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P01-C05-autograd.ipynb)

### Part 2: Introduction to Supervised Learning
* [1 - Linear Regression *(from scratch)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P02-C01-linear-regression-scratch.ipynb)
* [2 - Linear Regression *(with ``gluon``)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P02-C02-linear-regression-gluon.ipynb)
* [2.5 - Perceptron and SGD primer](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P02-C02.5-perceptron.ipynb)
* [3 - Multiclass Logistic Regression *(from scratch)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P02-C03-softmax-regression-scratch.ipynb)
* [4 - Multiclass Logistic Regression *(with ``gluon``)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P02-C04-softmax-regression-gluon.ipynb)
* [5 - Overfitting and regularization *(from scratch)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P02-C05-regularization-scratch.ipynb)
* ***Roadmap*** L1 and L2 Regularization (in ``gluon``)

### Part 3: Deep neural networks (DNNs) 
* [1 - Multilayer Perceptrons *(from scratch)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P03-C01-mlp-scratch.ipynb)
* [2 - Multilayer Perceptrons *(with ``gluon``)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P03-C02-mlp-gluon.ipynb)
* [3 - Dropout Regularization (from scratch)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P03-C03-mlp-dropout-scratch.ipynb)
* [4 - Dropout Regularization (with ``gluon``)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P03-C04-mlp-dropout-gluon.ipynb)

### Part 3.5: ``gluon`` Plumbing
* [1 - Introduction to ``gluon.Block`` and ``gluon.nn.Sequential()``](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P03.5-C01-plumbing.ipynb)
* [2 - Writing custom layers with ``gluon.Block``](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P03.5-C02-custom-layer.ipynb)
* [3 - Serialization: saving and loading models](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P03.5-C03-serialization.ipynb)
* Advanced Data IO
* Debugging your neural networks

### Part 4: Convolutional neural networks (CNNs) 
* [1 - Convolutional Neural Network *(from scratch!)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P04-C01-cnn-scratch.ipynb)
* [2 - Convolutional Neural Network *(with ``gluon``!)*](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P04-C02-cnn-gluon.ipynb)
* [3 - Introduction to Deep CNNs (AlexNet)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P04-C03-deep-cnns-alexnet.ipynb)
* [4 - Very deep networks and repeating blocks (VGG network)](./P04-C04-very-deep-nets-vgg.ipynb)
* ***Roadmap*** Batch Normalization (from scratch)
* ***Roadmap*** Batch Normalization (from with ``gluon``)

### Part 5: Recurrent neural networks (RNNs)
* [1 - Simple RNNs (from scratch)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P05-C01-simple-rnn.ipynb)
* [2 - LSTMS RNNs (from scratch)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P05-C02-lstm-scratch.ipynb)
* [3 - GRUs (from scratch)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P05-C03-gru-scratch.ipynb)
* [4 - RNNs (with ``gluon``)](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P05-C04-rnns-gluon.ipynb)
* ***Roadmap*** Dropout for recurrent nets
* ***Roadmap*** Zoneout regularization

### Part 6: Computer vision (CV)
* ***Roadmap*** Network of networks (inception & co)
* ***Roadmap*** Residual networks
* [Object detection](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P06-C03-object-detection.ipynb)
* ***Roadmap*** Fully-convolutional networks
* ***Roadmap*** Siamese (conjoined?) networks
* ***Roadmap*** Embeddings (pairwise and triplet losses)
* ***Roadmap*** Inceptionism / visualizing feature detectors
* ***Roadmap*** Style transfer
* [Fine-tuning](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P06-C09-fine-tuning.ipynb)

### Part 7: Natural language processing (NLP)
* ***Roadmap*** Word embeddings (Word2Vec)
* ***Roadmap*** Sentence embeddings (SkipThought)
* ***Roadmap*** Sentiment analysis
* ***Roadmap*** Sequence-to-sequence learning (machine translation)
* ***Roadmap*** Sequence transduction with attention (machine translation)
* ***Roadmap*** Named entity recognition 
* ***Roadmap*** Image captioning
* [Tree-LSTM for semantic relatedness](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P07-C08-tree-lstm.ipynb)
 
### Part 8: Unsupervised Learning
* ***Roadmap*** Introduction to autoencoders
* ***Roadmap*** Convolutional autoencoders (introduce upconvolution)
* ***Roadmap*** Denoising autoencoders
* ***Roadmap*** Variational autoencoders
* ***Roadmap*** Clustering

### Part 9: Adversarial learning
* ***Roadmap*** Two Sample Tests
* ***Roadmap*** Finding adversarial examples
* ***Roadmap*** Adversarial training

### Part 10: Generative adversarial networks (GANs)
* 1 - [Introduction to GANs](./P10-C01-gan-intro.ipynb)
* ***Roadmap*** DCGAN
* ***Roadmap*** Wasserstein-GANs
* ***Roadmap*** Energy-based GANS
* ***Roadmap*** Conditional GANs
* ***Roadmap*** Image transduction GANs (Pix2Pix)
* ***Roadmap*** Learning from Synthetic and Unsupervised Images 

### Part 11: Deep reinforcement learning (DRL)
* ***Roadmap*** Introduction to reinforcement learning
* ***Roadmap*** Deep contextual bandits
* ***Roadmap*** Deep Q-networks
* ***Roadmap*** Policy gradient
* ***Roadmap*** Actor-critic gradient

### Part 12: Variational methods and uncertainty
* ***Roadmap*** Dropout-based uncertainty estimation (BALD)
* ***Roadmap*** Weight uncertainty (Bayes-by-backprop)
* ***Roadmap*** Variational autoencoders

### Part 13: Optimization
* ***Roadmap*** SGD
* ***Roadmap*** Momentum
* ***Roadmap*** AdaGrad
* ***Roadmap*** RMSProp
* ***Roadmap*** Adam 
* ***Roadmap*** AdaDelta
* ***Roadmap*** SGLD / SGHNT

### Part 14: Optimization, Distributed and high-performance learning
* ***Roadmap*** Distributed optimization (Asynchronous SGD, ...)
* [Training with Multiple GPUs](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P14-C02-multiple-gpus-scratch.ipynb) 
* [Fast & flexible: combining imperative & symbolic nets with HybridBlocks](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/P14-C05-hybridize.ipynb)
* ***Roadmap*** Training with Multiple Machines
* ***Roadmap*** Combining imperative deep learning with symbolic graphs

### Part 15: Hacking MXNet
* ***Custom Operators***
* ...

### Part 16: Audio Processing
* ***Roadmap*** Intro to automatic speech recognition
* ***Roadmap*** Connectionist temporal classification (CSC) for unaligned sequences
* ***Roadmap*** Combining static and sequential data

### Part 17: Recommender systems
* ***Roadmap*** Latent factor models
* ***Roadmap*** Deep latent factor models
* ***Roadmap*** Bilinear models
* ***Roadmap*** Learning from implicit feedback

### Part 18: Time series
* ***Roadmap*** Forecasting
* ***Roadmap*** Modeling missing data
* ***Roadmap*** Combining static and sequential data

### Part 19 Tensor Methods
* ***Roadmap*** Introduction to tensor algebra
* ***Roadmap*** Tensor decomposition
* ***Roadmap*** Tensorized neural networks

### Appendix 1: Cheatsheets
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

In creating these tutorials, I have drawn inspiration from some the resources that allow me to learn machine learning and how to program with Theano and PyTorch:

* [Soumith Chintala's *Deep Learning with PyTorch: A 60 Minute Blitz*](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
* [Alec Radford's *Bare-bones intro to Theano*](https://github.com/Newmu/Theano-Tutorials) 
* [Video of Alec's intro to deep learning with Theano](https://www.youtube.com/watch?v=S75EdAcXHKk)
* [Chris Bishop's *Pattern Recognition and Machine Learning*](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738)

## Contribute
* Already, in the short time this project has been off the ground, we've gotten some helpful PRs from the community with pedagogical suggestions, typo corrections, and other useful fixes. If you're inclined, please contribute! 
