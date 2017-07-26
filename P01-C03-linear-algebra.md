# Linear Algebra

Now that you can store and manipulate data, let's briefly review the subset of
basic linear algebra that you'll need to understand most of the models. We'll
introduce all the basic concepts, the corresponding mathematical notaiton, and
their realization in code all in one place. If you're already confident basic
linear algebra, free to skim or skip this chapter.

```{.python .input  n=3}
import mxnet as mx
import mxnet.ndarray as nd
```

## Scalars

If you never studied linear algebra or machine learning, you're probably used to
working with single numbers, like $42.0$ and know how to do basic things like
add them together, multiply them. In mathematical notation, we'll represent
salars with ordinary lower cased letters ($x$, $y$, $z$). In MXNet, we can work
with scalars by creating NDArrays with just one element.

```{.python .input  n=4}
x = nd.array([3.0]) 
y = nd.array([2.0])
print(x + y)
print(x * y)
print(x / y)
print(nd.power(x,y))
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 5.]\n<NDArray 1 @cpu(0)>\n\n[ 6.]\n<NDArray 1 @cpu(0)>\n\n[ 1.5]\n<NDArray 1 @cpu(0)>\n\n[ 9.]\n<NDArray 1 @cpu(0)>\n"
 }
]
```

We can convert NDArrays to Python floats by calling their ``.asscalar()

```{.python .input  n=5}
x.asscalar()
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "3.0"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Vectors
You can think of vectors are simply a list of numbers ([1.0,3.0,4.0,2.0]). A
vector could represent numerical features of some real-world person or object,
like the last-record measurements across various vital signs for a patient in
the hospital. In math notation, we'll always denote vectors as bold-faced lower-
cased letters ($\boldsymbol{u}$, $\boldsymbol{v}$, $\boldsymbol{w})$. In MXNet,
we work with vectors via 1D NDArrays with an arbitrary number of components.

```{.python .input  n=6}
u = nd.zeros(shape=10)
v = nd.ones(shape=10)
print(u)
print(v)
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]\n<NDArray 10 @cpu(0)>\n\n[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]\n<NDArray 10 @cpu(0)>\n"
 }
]
```

We can refer to any element of a vector by using a subscript. For example, we
can refer to the $4$th element of $\boldsymbol{u}$ by $u_4$. Note that the
element $u_4$ is a scalar, so we don't bold-face the font when referring to it.

## Matrices

Just as vectors are an extension of scalars from 0 to 1 dimension, matrices
generalization vectors to two dimensions. Matrices, which we'll denote with
capital letters ($A$, $B$, $C$) are 2D arrays.

```{.python .input  n=7}
A = nd.random_normal(shape=(5,4))
B = nd.random_normal(shape=(5,4))
print(A)
print(B)
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 2.21220636  1.16307867  0.7740038   0.48380461]\n [ 1.04344046  0.29956347  1.18392551  0.15302546]\n [ 1.89171135 -1.16881478 -1.23474145  1.55807114]\n [-1.771029   -0.54594457 -0.45138445 -2.35562968]\n [ 0.57938355  0.54144019 -1.85608196  2.67850661]]\n<NDArray 5x4 @cpu(0)>\n\n[[-1.9768796   1.25463438 -0.20801921 -0.54877394]\n [ 0.2444218  -0.68106437 -0.03716067 -0.13531584]\n [-0.48774993  0.37723127 -0.02261727  0.41016445]\n [ 0.57461417  0.5712682   1.4661262  -2.7579627 ]\n [ 0.68629038  1.07628     0.35496104 -0.61413258]]\n<NDArray 5x4 @cpu(0)>\n"
 }
]
```

Matrices are useful data structures, they allow us to organize data that has
different modalities of variation. For example, returning to the example of
medical data, rows in our matrix might correspond to different patients, while
columns might correspond to different attributes.

We can access the scalar elements $a_{ij}$ of a matrix A by specifying the
indices for the row ($i$) and column ($j$) respectively. Let's grab the element
$a_{2,3}$ from the random matrix we initialized above.

```{.python .input  n=8}
A[2,3]
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "\n[ 1.55807114]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

We can also grab the vectors corresponding to entire rows $\boldsymbol{a}_{i,:}$
or columns $\boldsymbol{a}_{:,j}$.

```{.python .input  n=9}
print(A[2,:])
print(A[:,3])
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 1.89171135 -1.16881478 -1.23474145  1.55807114]\n<NDArray 4 @cpu(0)>\n\n[ 0.48380461  0.15302546  1.55807114 -2.35562968  2.67850661]\n<NDArray 5 @cpu(0)>\n"
 }
]
```

## Tensors

Just as vectors generalize scalars, and matrices generalize vectors, we can
actually build data structures with even more axes. Tensors, give us a generic
way of discussing arrays with an arbitrary number of axes. Vectors, for example
are be first-order tensors, and matrices are second-order tensors.

We'll have to think will become more important when we start working with
images, which arrive as 3D data structures, with axes corresponding to the
height, width, and the three (RGB) color channels. But in this chapter, we're
going to skip past and make sure you know the basics.

## Element-wise operations

Oftentimes, we want to perform element-wise operations. This means that we
perform a scalar operation on the corresponding elements of two vectors. So
given any two vectors $\boldsymbol{u}$ and $\boldsymbol{v}$ *of the same shape*,
and a scalar function $f$, we can perform the operation  we produce vector
$\boldsymbol{c} = f(\boldsymbol{u},\boldsymbol{v})$ by setting $c_i \gets f(u_i,
v_i)$. In MXNet, calling any of the standard arithmetic operators
(+,-,/,\*,\*\*) will invoke an elementwise operation.

```{.python .input  n=10}
print(u)
print(v) 
print(u + v)
print(u - v)
print(u * v)
print(u / v)
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]\n<NDArray 10 @cpu(0)>\n\n[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]\n<NDArray 10 @cpu(0)>\n\n[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]\n<NDArray 10 @cpu(0)>\n\n[-1. -1. -1. -1. -1. -1. -1. -1. -1. -1.]\n<NDArray 10 @cpu(0)>\n\n[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]\n<NDArray 10 @cpu(0)>\n\n[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]\n<NDArray 10 @cpu(0)>\n"
 }
]
```

We can call element-wise operations on any two tensors of the same shape,
including matrices.

```{.python .input  n=11}
print(A + B)
print(A[0,0] + B[0,0])
```

```{.json .output n=11}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.23532677  2.41771317  0.56598461 -0.06496933]\n [ 1.2878623  -0.3815009   1.14676487  0.01770963]\n [ 1.40396142 -0.79158354 -1.25735867  1.96823561]\n [-1.19641483  0.02532363  1.01474178 -5.11359215]\n [ 1.26567388  1.61772013 -1.50112092  2.06437397]]\n<NDArray 5x4 @cpu(0)>\n\n[ 0.23532677]\n<NDArray 1 @cpu(0)>\n"
 }
]
```

## Sums and means

The next more sophisticated thing we can do with arbitrary tensors is to
calculate the sum of their elements. In mathematical notation, we express sums
using the $\sum$ symbol. To express the sum of the elements in a vector
$\boldsymbol{u}$ of length $d$, we can write $\sum_{i=1}^d u_i$. In code, we can
just call ``nd.sum()``.

```{.python .input  n=12}
print(nd.sum(u))
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 0.]\n<NDArray 1 @cpu(0)>\n"
 }
]
```

We can similarly express sums over the elements of tensors of arbitrary shape.
For example, the sum of the elements of an $m \times n$ matrix A could be
written $\sum_{i=1}^{m} \sum{j=1}^{n} a_{i,j}$.

```{.python .input  n=13}
print(nd.sum(A))
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 5.17853642]\n<NDArray 1 @cpu(0)>\n"
 }
]
```

A related quantity to the sum is the *mean*, also commonly called the *average*.
We calculate the mean by dividing the sum by the total number of elements. With
mathematical notation, we could write the average over a vector
${\boldsymbol{u}$ as \frac{1}{d} \sum_{i=1}^{d} u_i$ and the average over a
matrix $A$ as  $\frac{1}{n \cdot m} \sum_{i=1}^{m} \sum_{j=1}^{n} a_{i,j}$. In
code, we could just call ``nd.mean()`` tensors of arbitrary shape:

```{.python .input  n=14}
print(nd.mean(u))
print(nd.mean(A))
```

```{.json .output n=14}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 0.]\n<NDArray 1 @cpu(0)>\n\n[ 0.25892681]\n<NDArray 1 @cpu(0)>\n"
 }
]
```

## Dot products

<!-- So far, we've only performed element-wise operations, sums and averages.
And if this was we could do, linear algebra probably wouldn't deserve it's own
chapter. However, -->

One of the most fundamental operations is the dot product. Given two vectors
$\boldsymbol{u}$ and $\boldsymbol{v}$, the dot product $\boldsymbol{u}^T \cdot
\boldsymbol{v}$ is a sum over the products of the corresponding elements:
$\boldsymbol{u}^T \cdot \boldsymbol{v} = \sum_{i=1}^{d} u_i \cdot v_i$.

```{.python .input  n=15}
u = nd.arange(0,5,1.)
v = nd.flip(nd.arange(0,5,1.), 0)
print(u)
print(v)
print(nd.dot(u,v))
```

```{.json .output n=15}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 0.  1.  2.  3.  4.]\n<NDArray 5 @cpu(0)>\n\n[ 4.  3.  2.  1.  0.]\n<NDArray 5 @cpu(0)>\n\n[ 10.]\n<NDArray 1 @cpu(0)>\n"
 }
]
```

Note that we can code the dot product over two vectors ``nd.dot(u, v)``
equivalently by performing an element-wise multiplication and then a sum:

```{.python .input  n=16}
nd.sum(u * v)
```

```{.json .output n=16}
[
 {
  "data": {
   "text/plain": "\n[ 10.]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 16,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Dot products are useful in a wide range of contexts. For example, given a set of
weights $\boldsymbol{w}$, the weighted sum of some values ${u}$ could be
expressed as the dot product $\boldsymbol{u}^T \boldsymbol{w}$. When the weights
are non-negative and sum to one ($\sum_{i=1}^{d} {w_i} = 1$), the dot product
expresses a *weighted average*. When two vectors each have length one (we'll
discuss what *length* means below in the section on norms), dot products can
also capture the cosine of the angle between two vectors.

## Matrix-vector products

Now that we know how to calculate dot products we can begin to understand
matrix-vector products.

```{.python .input}

```

```{.python .input}

```

## Matrix-matrix multiplication

```{.python .input}

```

```{.python .input}

```

## Norms

```{.python .input}

```

```{.python .input}

```
