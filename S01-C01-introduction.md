# Introduction

If you're a reasonable person, you might ask what is *mxnet-the-straight-dope* and why does it have such a strange name? 
Speaking to the former question, *mxnet-the-straight-dope* is an attempt to create a new kind of educational resource for deep learning. Our goal is to leverage the strengths of jupyter notebooks to present prose, graphics, equations, and (importantly) code together in one place. If we stick it out, we hope to cultivate here a resource that could be simultaneously a book, course material, a prop for live tutorials, and a resource for plagiarising (with our blessing) useful code. 

Regarding the stupid name, we are cognizant that the machine learning and the ecosystem have been thrust into an absurd place. In the early 2000s, comparatively few tasks in machine learning had been conquered, but we felt that we understood *how* and *why* those models worked (with some caveats) from first principles. By contrast, today's machine learning systems are extremely powerful, but we're not always sure precisely why.  

This new world offers enormous opportunity, but has also given rise to all manner of buffoonery. Research preprints have been flooded by clickbait, charlatans have sold AI startups, and the blogosphere is flooded with thought leadership pieces written by marketers bereft of any technical knowledge. Amid the chaos, easy money, and lax standards, we believe it's important not to take our models or the environment in which they are worshiped too seriously. Also, in order to both explain, visualize, and code the full breadth of models that we aim to address, it's important that the authors do not get bored while writing. 

## Organization

At present, we're aiming for the following format: But for a few (optional) notebooks providing a crashcourse in the basic mathematical background, each subsequent notebook will both

1. Introduce a reasonable number (perhaps one) of new concepts
2. Provide a single self-contained working example, using a real dataset

Throughout we'll be working with the MXNet library, which has the rare property of being flexible enough for research while being fast enough for production. We'll generally be using mxnet's new high-level imperative interface ``gluon``. Note that this is not the same as ``mxnet.module``, an older, symbolic interface supported by MXNet. 

We'll be teaching deep learning concepts from scratch. Sometimes, we'll want to delve into fine details about the models that are hidden from the user by ``gluon``'s advanced features. This comes up especialy in the basic tutorials, where we'll want you to understand everything that happens in a given layer. In these cases, we'll generally present two versions of the tutorial: one where we implement everything from scratch, relying only on NDArray and automatic differentiation, and another where we show how to do things succinctly with ``gluon``. Once we've taught you how a layer works, we can just use the ``gluon`` version in subsequent tutorials.



