# Deep convolutional neural networks 

In the previous chapters, you got a sense 
for how to classify images with convolutional neural network (CNNs). 
Specifically, we implemented a CNN with two convolutional layers 
interleaved with pooling layers, a singly fully-connected hidden layer,
and a softmax output layer. 
That architecture loosely resembles a neural network affectionately named LeNet,
in honor [Yann LeCun](http://yann.lecun.com/),
an early pionneer of convolutional neural networks and the first 
to [reduced them to practice in 1989](http://www.mitpressjournals.org/doi/abs/10.1162/neco.1989.1.4.541) 
by training them with gradient descent (i.e. backpropagation).
At the time, this was fairly novel idea.
A cadre of reserchers interested in biologically-inspired learning models had taken to investigating artificial simulations of neurons as learning models. 
However, as remains true to this day, few researchers believed that real brains learn by gradient descent.
The community of neural networks researchers had explored many other learning rules.
LeCun demonstrated that CNNs trained by gradient descent,
could get state-of-the-art results on the task of recognizing hand-written digits. 
These groundbreaking results put CNNs on the map.

However, in the intervening years, neural networks were superceded by a numerous other methods.
Neural networks were considered slow to train, 
and there wasn't wide consensus on whether it was possible 
to train very deep neural networks from a random initialization of the weights. 
Moreover, training networks with many channels, layers, 
and parameters were required excessive computation 
relative to the resources available decades ago. 
While it was possible to train a LeNet for MNIST digit classification and get good scores,
neural networks fell out of favor on larger, real-world datasets. 

Instead researchers precomputed features based on a mixture of elbow grease, 
knowledge of optics, and black magic. 
A typical pattern was this: 
1. Grab a cool dataset
2. Preprocess it with giant bag of predetermined feature functions
3. Dump the representations into a simple linear model to do the *machine learning part*. 

This was the state of affairs in computer vision up until 2012, 
just before deep learning began to change the world of applied machine learning. 
One of us (Zack) entered graduate school in 2013. 
A friend in graduate school summarized the state of affairs thusly:

If you spoke to machine learning researchers, 
they believed that machine learning was both important and beautiful.
Elegant theories proved the properties of various classifiers.
The field of machine learning was thriving, rigorous and eminently useful.
However, if you spoke to a comuter vision researcher, you'd hear a very different story.
The dirty truth of image recognition, they'd tell you,
is that the really important aspects of the ML for CV pipeline were data and features.
A slightly cleaner dataset, or a slightly better hand-tuned feature mattered a lot to the final accuracy.
However, the specific choice of classifier was little more than an afterthought.
At the end of the day you could throw your features in a logistic regression model, a support vector machine, or any other classifier of choice, and they would all perform roughly the same. 






## Learning the representations

Another way to cast the state of affairs is that fixing a dataset, 
the most important part of the pipeline was the representation.
And up until 2012, this part was done mechanically, based on some hard-fought intuition.
In fact, engineering a new set of feture functions, improving results, and writing up the method was a prominent genre of paper.

Another group of researchers had other plans. They believed that features themselves ought to be learned. 
Moreover they believed that to be reasonable complex, the fetures ought to be hierarchically composed.
These researchers, including Yann LeCun, Geoff Hinton, Yoshua Bengio, Andrew Ng, Shun-ichi Amari, and Juergen Schmidhuber believed that by jointly training many layers of a neural network, they might come to learn hierarchical representations of data. 
In the case of an image, the lowest layers might come to detect edges, colors, and textures. 

![](./img/filters.png)


Higher layers might build upon these representations to represent larger structures, like eyes, noses, blades of grass, and features. 
Yet higher layers might represent whole objects like people, airplanes, dogs, or frisbees. 
And ultimately, before the classification layer, the final hidden state might represent a compact representation of the image that summarized the contents in a space where data belonging to different categories would be linearly seperable. For years, researchers 

## Missing ingredient 1: data 

Despite the sustained interest of a committed group of researchers in learning deep representations of visual data, for a long time these ambitions were frustrated. The failures to make progress owed to a few factors. First, while this wasn't yet known, supervised deep models with many representation require large amounts of labeled training data in order to out perform classical approaches. However, given the limited storage capacity of computers and the comparatively tighter research budgets in the 1990s and prior, most research relied on tiny datasets. For example, most many credible research papers relied on a small set of corpora hosted by UCI, many of which contained hundreds or a few thousand images.

This changed in a big way when Fei-Fei Li presented the ImageNet database in 2009. The ImageNet dataset dwarved all previous research datasets. It contained one million images: one thousand each from one thousand distinct classes. 

![](./img/imagenet.jpeg)

This dataset pushed both computer vision and machine learning research into a new regime where the previous best methods would no longer dominate. 

## Missing ingredient 2: hardware

Deep Learning has a rapacious need for computation. This is one of the main reasons why in the 90s and early 2000s algorithms based on convex optimization were the preferred way of solving problems. After all, convex algorithms have fast rates of convergence, global minima, and efficient algorithms can be found. 

The game changer was the availability of GPUs. They had longtime been tuned for graphics processing in computer games. In particular, they were optimized for high throughput 4x4 matrix-vector products, since these are needed for many computer graphics tasks. Fortunately, the math required for that is very similar to convolutional layers in deep networks. Likewise, around that time, NVIDIA and ATI had begun optimizing GPUs for general compute operations, going as far as renaming them GPGPU (General Purpose GPUs). 

To provide some intuition, consider the cores of a modern microprocessor. Each of the cores is quite powerful, running at a high clock frequency, it has quite advanced and large caches (up to several MB of L3). It is very good at executing a very wide range of code, with branch predictors, a deep pipeline and lots of other things that make it great at executing regular programs. This apparent strength, however, is also its Achilles heel: general purpose cores are very expensive to build. They require lots of chip area, a sophisticated support structure (memory interfaces, caching logic between cores, high speed interconnects, etc.), and they're comparatively bad at any single task. Modern laptops have up to 4 cores, and even high end servers rarely exceed 64 cores, simply because it is not cost effective. 

Compare that with GPUs. They consist of 100-1000 small processing elements (the details differ somewhat betwen NVIDIA, ATI, ARM and other chip vendors), often grouped into larger groups (NVIDIA calls them warps). While each of them is relatively weak, running at sub-1GHz clock frequency, it is the total number of such cores that makes them orders of magnitude faster than CPUs. For instance, NVIDIA's latest Volta generation offers up to 120 TFlops per chip for specialized instructions (and up to 24 TFlops for more general purpose ones), while floating point performance of CPUs has not exceeded 1 TFlop to date. The reason for why this is possible is actually quite simple: firstly, power consumption tends to grow *quadratically* with clock frequency. Hence, for the power budget of a CPU core that runs 4x faster (a typical number) you can use 16 GPU cores at 1/4 the speed, which yields 16 x 1/4 = 4x the performance. Furthermore GPU cores are much simpler (in fact, for a long time they weren't even *able* to execute general purpose code), which makes them more energy efficient. Lastly, many operations in deep learning require high memory bandwidth. Again, GPUs shine here with buses that are at least 10x as wide as many CPUs. 

Back to 2012. A major breakthrough came when Alex Krizhevsky and Ilya Sutskever 
implemented a deep convolutional neural network that could run on GPU hardware. They realized that 
the computational bottlenecks in CNNs (convolutions and matrix multiplications) are all operatiosn that could be parallelized in hardware. Using two NIVIDA GTX 580s with 3GB of memory (depicted below) they implemented fast convolutions. The code [cuda-convnet](https://code.google.com/archive/p/cuda-convnet/) was good enough that for several years it the industry standard and powered the first couple years of the deep learning boom.  

![](./img/gtx-580-gpu.jpeg)


## AlexNet

In 2012, using their cuda-convnet implementation on an eight-layer CNN, 
Khrizhevsky, Sutskever and Hinton won the ImageNet challenge on image recognition by a wide margin.
Their model, [introduced in this paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). It is *very* similar to the LeNet architecture from 1995. 

In the rest of the chapter we're going to implement a similar model to the one designed them. Due to memory constraints on the GPU they did some wacky things to make the model fit. For example, they designed a dual-stream architecture in which half of the nodes live on each GPU. The two streams, and thus the two GPUs only communicate at certain layers. This limits the amount of overhead for keeping the two GPUs in sync with each other. Fortunately, distributed deep learning has advanced a long way in the last few years, so we won't be needing those features (unless for very unusual architectures). In later sections, we'll go into greater depth on how you can speed up your networks by training on many GPUs (in AWS you can get up to 16 on a single machine with 12GB each), and how you can train on many machine simultaneously. As usual, we'll start by importing the same dependencies as in the past gluon tutorials:

```{.python .input}
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
mx.random.seed(1)
```

```{.python .input}
ctx = mx.gpu()
```

## Load up a dataset

Now let's load up a dataset. This time we're going to use gluon's new `vision` package, and import the CIFAR dataset. Cifar is a much smaller color dataset, roughly the dimensions of ImageNet. It contains 50,000 training and 10,000 test images. The images belong in equal quantities to 10 categories. While this dataset is considerably smaller than the 1M image, 1k category, 256x256 imagenet dataset, we'll use it here to demonstrate the model because we don't want to assume that you have a license to the dataset or a machine that can store it comfortably. To give you some sense for the proportions of working with ImageNet data, we'll upsample the images to 224x224 (the size used in the original AlexNet.  

```{.python .input}
def transformer(data, label):
    data = mx.image.imresize(data, 224, 224)
    data = mx.nd.transpose(data, (2,0,1))
    data = data.astype(np.float32)
    return data, label

```

```{.python .input}
batch_size = 64
train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10('./data', train=True, transform=transformer),
    batch_size=batch_size, shuffle=True, last_batch='discard')

test_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10('./data', train=False, transform=transformer),
    batch_size=batch_size, shuffle=False, last_batch='discard')
```

```{.python .input}
for d, l in train_data:
    break
```

```{.python .input}
print(d.shape, l.shape)
```

```{.python .input}
d.dtype
```

## The AlexNet architecture

This model has some notable features. 
First, in contrast to the relatively tiny LeNet, 
AlexNet contains 8 layers of transformations,
five convolutional layers followed by two fully connected hidden layers and an output layer.

The convolutional kernels in the first convolutional layer are reasonably large at $11 \times 11$, in the second  they are $5\times5$ and thereafter they are $3\times3$. Moreover, the first, second, and fifth convolutional layers are each followed by overlapping pooling operations with pool size $3\times3$ and stride ($2\times2$). 

Following the convolutional layers, the original AlexNet had fully-connected layers with 4096 nodes each. Using `gluon.nn.Sequential()`, we can define the entire AlexNet architecture in just 14 lines of code.  Besides the specific architectural choices and the data preparation, we can recycle all of the code we'd used for LeNet verbatim. 

[**right now relying on a different data pipeline (the new gluon.vision). Sync this with the other chapter soon and commit to one data pipeline.**]

[add dropout once we are 100% final on API]

```{.python .input}
alex_net = gluon.nn.Sequential()
with alex_net.name_scope():
    #  First convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=96, kernel_size=11, strides=(4,4), activation='relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))    
    #  Second convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=192, kernel_size=5, activation='relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=(2,2)))            
    # Third convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, activation='relu'))
    # Fourth convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=384, kernel_size=3, activation='relu')) 
    # Fifth convolutional layer
    alex_net.add(gluon.nn.Conv2D(channels=256, kernel_size=3, activation='relu'))
    alex_net.add(gluon.nn.MaxPool2D(pool_size=3, strides=2))    
    # Flatten and apply fullly connected layers
    alex_net.add(gluon.nn.Flatten())
    alex_net.add(gluon.nn.Dense(4096, activation="relu"))
    alex_net.add(gluon.nn.Dense(4096, activation="relu"))
    alex_net.add(gluon.nn.Dense(10))
```

## Initialize parameters

```{.python .input}
alex_net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
```

## Optimizer

```{.python .input}
trainer = gluon.Trainer(alex_net.collect_params(), 'sgd', {'learning_rate': .001})
```

## Softmax cross-entropy loss

```{.python .input}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Evaluation loop

```{.python .input}
def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for d, l in data_iterator:
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]
```

## Training loop

```{.python .input}
###########################
#  Only one epoch so tests can run quickly, increase this variable to actually run
###########################
epochs = 1
smoothing_constant = .01


for e in range(epochs):
    for i, (d, l) in enumerate(train_data):
        data = d.as_in_context(ctx)
        label = l.as_in_context(ctx)
        with autograd.record():
            output = alex_net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
        
        ##########################
        #  Keep a moving average of the losses
        ##########################
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0)) 
                       else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)
            
    test_accuracy = evaluate_accuracy(test_data, alex_net)
    train_accuracy = evaluate_accuracy(train_data, alex_net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))    
```

```{.python .input}

```
