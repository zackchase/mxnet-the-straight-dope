# Manipulate data the MXNet way with `ndarray`

It's impossible to get anything done if we can't manipulate data. This has two parts - loading data and processing data once it's inside the computer. This notebook is about the latter. So let's start by introducing NDArrays, MXNet's primary tool for storing and transforming data. If you've worked with NumPy before, you'll notice that NDArrays are by design similar to NumPy's multi-dimensional array. However, they confer a few key advantages. First, NDArrays support asynchronous computation on CPU, GPU, and distributed cloud architectures. Second, they provide support for automatic differentiation. These properties make NDArray an ideal library for machine learning, both for researchers and engineers launching production systems.


Refer to [prob](P01-C03-linear-algebra.md)

## Getting started

In this chapter, we'll get you going with the basic functionality. Don't worry if you don't understand any of the basic math, like element-wise operations or normal distributions. In the next two chapters we'll take another pass at NDArray, teaching you both the math you'll need and how to realize it in code.

To get started, let's import `mxnet`. We'll also import `ndarray` from `mxnet` for convenience. We’ll make a habit of setting a random seed so that you always get the same results that we do.

```{.python .input}
import mxnet as mx
from mxnet import nd
mx.random.seed(1)
```

Next, let's see how to create an NDArray, without any values initialized. Specifically, we'll create a 2D array (also called a *matrix*) with 3 rows and 4 columns.

```{.python .input}
x = nd.empty((3, 4))
print(x)
```

The `empty` method just grabs some memory and hands us back a matrix without setting the values of any of its entries. This means that the entries can have any form of values, including very big ones! But typically, we'll want our matrices initialized. Commonly, we want a matrix of all zeros.

```{.python .input}
x = nd.zeros((3, 5))
x
```

Similarly, `ndarray` has a function to create a matrix of all ones.

```{.python .input}
x = nd.ones((3, 4))
x
```

Often, we'll want to create arrays whose values are sampled randomly. This is especially common when we intend to use the array as a parameter in a neural network. In this snippet, we initialize with values drawn from a standard normal distribution with zero mean and unit variance.

```{.python .input}
y = nd.random_normal(0, 1, shape=(3, 4))
y
```

As in NumPy, the dimensions of each NDArray are accessible via the `.shape` attribute.

```{.python .input}
y.shape
```

We can also query its size, which is equal to the product of the components of the shape. Together with the precision of the stored values, this tells us how much memory the array occupies.

```{.python .input}
y.size
```

## Operations

NDArray supports a large number of standard mathematical operations. Such as element-wise addition:

```{.python .input}
x + y
```

Multiplication:

```{.python .input}
x * y
```

And exponentiation:

```{.python .input}
nd.exp(y)
```

We can also grab a matrix's transpose to compute a proper matrix-matrix product.

```{.python .input}
nd.dot(x, y.T)
```

We'll explain these opoerations and present even more operators in the [linear algebra](P01-C03-linear-algebra.ipynb) chapter. But for now, we'll stick with the mechanics of working with NDArrays.

## In-place operations

In the previous example, every time we ran an operation, we allocated new memory to host its results. For example, if we write `y = x + y`, we will dereference the matrix that `y` used to point to and insted point it at the newly allocated memory. We can show this using Python's `id()` function, which tells us precisely which object a variable refers to.

```{.python .input}
print('id(y):', id(y))
y = y + x
print('id(y):', id(y))
```

We can assign the result to a previously allocated array with slice notation, e.g., `result[:] = ...`.

```{.python .input}
z = nd.zeros_like(x)
print('id(z):', id(z))
z[:] = x + y
print('id(z):', id(z))
```

However, `x+y` here will still allocate a temporary buffer to store the result before copying it to z. To make better use of memory, we can perform operations in place, avoiding temporary buffers. To do this we specify the `out` keyword argument every operator supports:

```{.python .input}
nd.elemwise_add(x, y, out=z)
```

If we're not planning to re-use ``x``, then we can assign the result to ``x`` itself. There are two ways to do this in MXNet.
1. By using slice notation x[:] = x op y
2. By using the op-equals operators like `+=`

```{.python .input}
print('id(x):', id(x))
x += y
x
print('id(x):', id(x))
```

Refer to [prob](P01-C03-linear-algebra.md)
