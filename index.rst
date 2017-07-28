Deep Learning - The Straight Dope
==================================

This project contains an incremental sequence of notebooks designed to make
learning machine learning, MXNet, and the ``gluon`` interface easy
(meta-easy?). Our goal is to leverage the strengths of Jupyter notebooks to
present prose, graphics, equations, and code together in one place. If we're
successful, the result will be a resource that could be simultaneously a book,
course material, a prop for live tutorials, and a resource for plagiarising
(with our blessing) useful code. To our knowledge there's no source out there
that teaches either (1) the full breadth of concepts in modern deep learning
or (2) interleaves an engaging textbook with runnable code. We'll find out by
the end of this venture whether or not that void exists for a good reason.

Throughout this book, we rely upon MXNet to teach core concepts, advanced
topics, and a full complement of applications. MXNet is widely used in
production environments owing to its strong reputation for speed. Now with
``gluon``, MXNet's new imperative interface (alpha), doing research in MXNet is
easy.

I've designed these tutorials so that you can traverse the curriculum in one of
three ways.

* Anarchist - Choose whatever you want to read, whenever you want to read it.
* Imperialist - Proceed throught the tutorials in order (1, 2, 3a, 3b, 4a, 4b, ...).
  In this fashion you will be exposed to each model first from scratch,
  writing all the code ourselves but for the basic linear algebra primitives and
  automatic differentiation.
* Capitalist - If you would like to specialize to either the raw interface or
  the high-level ``gluon`` front end choose either (1, 2, 3a, 4a, ...) or (1, 2,
  3b, 4b, ...) respectively.


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Crashcourse

   P01-*

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Introduction to Supervised Learning

   P02-*

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Deep Neural Networks

   P03-*

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Convolutional Neural Networks

   P04-*

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Recurrent Neural Networks

   P05-*

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Computer Vision

   P06-*

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: High-performance and distributed training

   P14-*


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Developer Documents

   docs/*
