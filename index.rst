Deep Learning - The Straight Dope
==================================

This repo contains an incremental sequence of notebooks designed to teach deep learning, `Apache MXNet (incubating) <https://github.com/apache/incubator-mxnet>`_, and the gluon interface. Our goal is to leverage the strengths of Jupyter notebooks to present prose, graphics, equations, and code together in one place. If we're successful, the result will be a resource that could be simultaneously a book, course material, a prop for live tutorials, and a resource for plagiarising (with our blessing) useful code. To our knowledge there's no source out there that teaches either (1) the full breadth of concepts in modern deep learning or (2) interleaves an engaging textbook with runnable code. We'll find out by the end of this venture whether or not that void exists for a good reason.

Another unique aspect of this book is its authorship process. We are developing this resource fully in the public view and are making it available for free in its entirety. While the book has a few primary authors to set the tone and shape the content, we welcome contributions from the community and hope to coauthor chapters and entire sections with experts and community members. Already we've received contributions spanning typo corrections through full working examples.


How to contribute
=================

To clone or contribute, visit `Deep Learning - The Straight Dope
<http://github.com/zackchase/mxnet-the-straight-dope>`_ on Github, and check our
`contribution guideline <A02-C02-contribute.md>`_.

Dependencies
============

To run these notebooks, a recent version of MXNet is required. The easiest way is to install the nightly build MXNet through ``pip``. E.g.::

    $ pip install mxnet --pre --user

More detailed instructions are available `here <A02-C01-install.md>`_

.. toctree::
   :maxdepth: 1
   :caption: Crashcourse
   :numbered:

   preface
   introduction
   ndarray
   linear-algebra
   probability
   autograd
