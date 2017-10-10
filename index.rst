Deep Learning - The Straight Dope
==================================

This repo contains an incremental sequence of notebooks designed to teach deep learning, `Apache MXNet (incubating) <https://github.com/apache/incubator-mxnet>`_, and the gluon interface. Our goal is to leverage the strengths of Jupyter notebooks to present prose, graphics, equations, and code together in one place. If we're successful, the result will be a resource that could be simultaneously a book, course material, a prop for live tutorials, and a resource for plagiarising (with our blessing) useful code. To our knowledge there's no source out there that teaches either (1) the full breadth of concepts in modern deep learning or (2) interleaves an engaging textbook with runnable code. We'll find out by the end of this venture whether or not that void exists for a good reason.

Another unique aspect of this book is its authorship process. We are developing this resource fully in the public view and are making it available for free in its entirety. While the book has a few primary authors to set the tone and shape the content, we welcome contributions from the community and hope to coauthor chapters and entire sections with experts and community members. Already we've received contributions spanning typo corrections through full working examples. 


How to contribute
=================

To clone or contribute, visit `Deep Learning - The Straight Dope <http://github.com/zackchase/mxnet-the-straight-dope>`_ on Github.

Dependencies 
============

To run these notebooks, a recent version of MXNet is required. The easiest way is to install the nightly build MXNet through ``pip``. E.g.::

    $ pip install mxnet --pre --user
    
More detailed instructions are available `here <docs/C01-install.html>`_


Part 1: Deep Learning Fundamentals
==================================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Crash course
   
   chapter01_crashcourse/preface
   chapter01_crashcourse/introduction
   chapter01_crashcourse/ndarray
   chapter01_crashcourse/linear-algebra
   chapter01_crashcourse/probability
   chapter01_crashcourse/autograd


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Introduction to supervised learning

   chapter02_supervised-learning/linear-regression-scratch
   chapter02_supervised-learning/linear-regression-gluon
   chapter02_supervised-learning/perceptron
   chapter02_supervised-learning/softmax-regression-scratch
   chapter02_supervised-learning/softmax-regression-gluon
   chapter02_supervised-learning/regularization-scratch
   chapter02_supervised-learning/regularization-gluon
   chapter02_supervised-learning/environment

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Deep neural networks

   chapter03_deep-neural-networks/mlp-scratch
   chapter03_deep-neural-networks/mlp-gluon
   chapter03_deep-neural-networks/mlp-dropout-scratch
   chapter03_deep-neural-networks/mlp-dropout-gluon
   chapter03_deep-neural-networks/plumbing
   chapter03_deep-neural-networks/custom-layer
   chapter03_deep-neural-networks/serialization

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Convolutional neural networks

   chapter04_convolutional-neural-networks/cnn-scratch
   chapter04_convolutional-neural-networks/cnn-gluon
   chapter04_convolutional-neural-networks/deep-cnns-alexnet
   chapter04_convolutional-neural-networks/very-deep-nets-vgg
   chapter04_convolutional-neural-networks/cnn-batch-norm-scratch
   chapter04_convolutional-neural-networks/cnn-batch-norm-gluon

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Recurrent neural networks

   chapter05_recurrent-neural-networks/simple-rnn
   chapter05_recurrent-neural-networks/lstm-scratch
   chapter05_recurrent-neural-networks/gru-scratch
   chapter05_recurrent-neural-networks/rnns-gluon

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Optimization

   chapter06_optimization/optimization-intro
   chapter06_optimization/gd-sgd
   chapter06_optimization/sgd-momentum.ipynb

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: High-performance and distributed training

   chapter07_distributed-learning/hybridize
   chapter07_distributed-learning/multiple-gpus-scratch
   chapter07_distributed-learning/multiple-gpus-gluon
   chapter07_distributed-learning/training-with-multiple-machines
   

Part 2: Applications
====================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Computer vision

   chapter08_computer-vision/object-detection
   chapter08_computer-vision/fine-tuning
   chapter08_computer-vision/visual-question-answer 
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Natural language processing

   chapter09_natural-language-processing/tree-lstm

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Recommender systems

   chapter11_recommender-systems/intro-recommender-systems   

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Time series

   chapter12_time-series/lds-scratch   
   chapter12_time-series/issm-scratch

Part 3: Advanced Topics
=======================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Generative adversarial networks

   chapter14_generative-adversarial-networks/gan-intro
   chapter14_generative-adversarial-networks/dcgan
   chapter14_generative-adversarial-networks/pixel2pixel
   
   
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Variational methods

   chapter18_variational-methods-and-uncertainty/bayes-by-backprop.ipynb
   chapter18_variational-methods-and-uncertainty/bayes-by-backprop-gluon.ipynb


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Cheat sheets

   cheatsheets/kaggle-gluon-kfold.ipynb

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Developer documents

   docs/*
