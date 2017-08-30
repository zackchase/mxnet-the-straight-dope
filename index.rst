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
   :caption: Crashcourse
   chapter01_crashcourse/preface.ipynb
   chapter01_crashcourse/introduction.ipynb
   chapter01_crashcourse/ndarray.ipynb
   chapter01_crashcourse/linear-algebra.ipynb
   chapter01_crashcourse/probability.ipynb
   chapter01_crashcourse/autograd.ipynb


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Introduction to supervised learning

   chapter02_supervised-learning/linear-regression-scratch.ipynb
   chapter02_supervised-learning/linear-regression-gluon.ipynb
   chapter02_supervised-learning/perceptron.ipynb
   chapter02_supervised-learning/softmax-regression-scratch.ipynb
   chapter02_supervised-learning/softmax-regression-gluon.ipynb
   chapter02_supervised-learning/regularization-scratch.ipynb
   chapter02_supervised-learning/regularization-gluon.ipynb
   chapter02_supervised-learning/loss.ipynb
   chapter02_supervised-learning/environment.ipynb

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Deep neural networks

   chapter03_deep-neural-networks/mlp-scratch.ipynb
   chapter03_deep-neural-networks/mlp-gluon.ipynb
   chapter03_deep-neural-networks/mlp-dropout-scratch.ipynb
   chapter03_deep-neural-networks/mlp-dropout-gluon.ipynb
   chapter03_deep-neural-networks/plumbing.ipynb
   chapter03_deep-neural-networks/custom-layer.ipynb
   chapter03_deep-neural-networks/serialization.ipynb

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Convolutional neural networks

   chapter04_convolutional-neural-networks/cnn-scratch.ipynb
   chapter04_convolutional-neural-networks/cnn-gluon.ipynb
   chapter04_convolutional-neural-networks/deep-cnns-alexnet.ipynb
   chapter04_convolutional-neural-networks/very-deep-nets-vgg.ipynb
   chapter04_convolutional-neural-networks/cnn-batch-norm-scratch.ipynb
   chapter04_convolutional-neural-networks/cnn-batch-norm-gluon.ipynb

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Recurrent neural networks

   chapter05_recurrent-neural-networks/simple-rnn.ipynb
   chapter05_recurrent-neural-networks/lstm-scratch.ipynb
   chapter05_recurrent-neural-networks/gru-scratch.ipynb
   chapter05_recurrent-neural-networks/rnns-gluon.ipynb

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Optimization

   chapter06_optimization/optimization-intro.ipynb
   chapter06_optimization/gd-sgd.ipynb

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: High-performance and distributed training

   hybridize.ipynb
   multiple-gpus-scratch.ipynb
   multiple-gpus-gluon.ipynb
   training-with-multi-machines.ipynb
   

Part 2: Applications
====================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Computer vision

   chapter08_computer-vision/object-detection.ipynb
   chapter08_computer-vision/fine-tuning.ipynb
   
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Computer vision

   chapter08_computer-vision/object-detection.ipynb
   chapter08_computer-vision/fine-tuning.ipynb   
   
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Natural language processing

   chapter09-natural-language-processing/tree-lstm.ipynb
   

Part 3: Advanced Topics
=======================

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Generative adversarial networks

   chapter14_generative-adversarial-networks/gan-intro.ipynb
   chapter14_generative-adversarial-networks/dcgan.ipynb

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Developer documents

   docs/*
