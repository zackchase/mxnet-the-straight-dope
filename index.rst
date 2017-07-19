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

A `PDF <TheStraightDope.pdf>`_ version is also available


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   1-ndarray
   2-autograd
   3a-linear-regression-scratch
   3b-linear-regression-gluon
   4a-softmax-regression-scratch
   4b-softmax-regression-gluon
   5a-mlp-scratch
   5b-mlp-gluon
   6a-cnn-scratch
   6b-cnn-gluon

.. toctree::
   :maxdepth: 1
   :caption: Developer Documents

   docs/install
   docs/contribute
