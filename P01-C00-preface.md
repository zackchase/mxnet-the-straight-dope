# Preface

If you're a reasonable person, you might ask, "what is *mxnet-the-straight-dope*?" You might also ask, "why does it have such an ostentatious name?" Speaking to the former question, *mxnet-the-straight-dope* is an attempt to create a new kind of educational resource for deep learning. Our goal is to leverage the strengths of Jupyter notebooks to present prose, graphics, equations, and (importantly) code together in one place. If we're successful, the result will be a resource that could be simultaneously a book, course material, a prop for live tutorials, and a resource for plagiarising (with our blessing) useful code. To our knowledge, few available resources aim to teach either (1) the full breadth of concepts in modern machine learning or (2) interleave an engaging textbook with runnable code. We'll find out by the end of this venture whether or not that void exists for a good reason.

Regarding the name, we are cognizant that the machine learning community and the ecosystem in which we operate have lurched into an absurd place. In the early 2000s, comparatively few tasks in machine learning had been conquered, but we felt that we understood *how* and *why* those models worked (with some caveats). By contrast, today's machine learning systems are extremely powerful, but huge open questions remains as to why precisely they are so effective.  

This new world offers enormous opportunity, but has also given rise to considerable buffoonery. Research preprints have been flooded by clickbait, AI startups have sometimes received overly optimistic valuations, and the blogosphere is flooded with thought leadership pieces written by marketers bereft of any technical knowledge. Amid the chaos, easy money, and lax standards, we believe it's important not to take our models or the environment in which they are worshiped too seriously. Also, in order to both explain, visualize, and code the full breadth of models that we aim to address, it's important that the authors do not get bored while writing. 

## Organization

At present, we're aiming for the following format: But for a few (optional) notebooks providing a crashcourse in the basic mathematical background, each subsequent notebook will both

1. Introduce a reasonable number (perhaps one) of new concepts
2. Provide a single self-contained working example, using a real dataset

This will be interleaved by background material, as needed. That is, we will often err on the side of making tools available before explaining them fully (and we will follow up by explaining the background later). For instance, we will use Stochastic Gradient Descent before fully explaining why it is useful. This helps with giving practitioners the necessary ammunition to solve problems quickly, at the expense of requiring the reader to trust us with some decisions, at least in the short term. 
Throughout we'll be working with the MXNet library, which has the rare property of being flexible enough for research while being fast enough for production. We'll generally be using MXNet's new high-level imperative interface ``gluon``. Note that this is not the same as ``mxnet.module``, an older, symbolic interface supported by MXNet. 

We'll be teaching deep learning concepts from scratch. Sometimes, we'll want to delve into fine details about the models that are hidden from the user by ``gluon``'s advanced features. This comes up especially in the basic tutorials, where we'll want you to understand everything that happens in a given layer. In these cases, we'll generally present two versions of the tutorial: one where we implement everything from scratch, relying only on NDArray and automatic differentiation, and another where we show how to do things succinctly with ``gluon``. Once we've taught you how a layer works, we can just use the ``gluon`` version in subsequent tutorials.

## Learning by doing

Many textbooks teach a series of topics, each in exhaustive detail. For example, Chris Bishop's excellent textbook, [Pattern Recognition and Machine Learning](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738) teaches each topic so thoroughly, that getting to the chapter on linear regression requires a non-trivial amount of work. When I was first learning machine learning, this actually limited the book's usefulness as an introductory text. When I rediscovered it a couple years later, I loved it precisely for its thoroughness but it's still not how I could imagine learning in the first place. 

Instead, in this book, we'll teach most concepts just in time. For the fundamental preliminaries like linear algebra and probability, we'll provide a brief crash course from the outset, but we want you to taste the satisfaction of training your first model before worrying about exotic probability distributions. 
