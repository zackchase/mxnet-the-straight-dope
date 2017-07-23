Much easy, so MXNet. Wow.
=========================

MXNet is widely used in production environments owing to its strong reputation
for speed. Now with ``gluon``, MXNet's new imperative interface (alpha), doing
research in MXNet is easy.

This project contains an incremental sequence of tutorials to make learning
gluon easy (meta-easy?). These tutorials are designed with public presentations
in mind and are composed as Jupyter notebooks where each notebook corresponds to
roughly 20 minutes of rambling and each codeblock could correspond to roughly
one slide.


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
   :caption: High-performance and distributed training

   P14-*

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Developer Documents

   docs/*
