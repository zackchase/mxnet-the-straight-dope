# Object Detection Using Convolutional Neural Networks

Object detection is a popular computer vision technology that deals with detecting instances of semantic objects (such as humans, animals, buildings and so on) in images or videos, which is a fundamental process to solve more advanced computer vision problems. 

We are going to present a minimal example using ``gluon`` to illustrate how Convolutional Neural Networks can learn to detect objects.

```{.python .input  n=1}
from __future__ import print_function, division
import mxnet as mx
from mxnet import nd
from mxnet import autograd as ag
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.contrib.ndarray import MultiBoxPrior, MultiBoxTarget, MultiBoxDetection
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time
```

## Prepare data

First of all, we need a toy dataset to play with. Tiny datasets for object detection are rare, however, we can create a dataset for detection task from MNIST!

```{.python .input  n=2}
mnist = mx.test_utils.get_mnist()
batch_size = 64
train_data = mx.io.NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size, shuffle=True)
test_data = mx.io.NDArrayIter(mnist["test_data"], mnist["test_label"], batch_size, shuffle=True)
class_names = [str(x) for x in range(10)]
num_class = 10
print('Class names:', class_names)
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Class names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']\n"
 }
]
```

Now we grab one batch, and print it to see the internal array shapes

```{.python .input  n=3}
train_data.reset()
batch = train_data.next()
print(batch)
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "DataBatch: data shapes: [(64, 1, 28, 28)] label shapes: [(64,)]\n"
 }
]
```

And we can display the sample image using `matplotlib`

```{.python .input  n=4}
demo = batch.data[0][0].asnumpy()  # grab the first image, convert to numpy array
demo = demo.transpose((1, 2, 0))  # we want channel to be the last dimension
plt.imshow(demo[:, :, (0, 0, 0)])  # convert to 3-channels and display
plt.show()
```

```{.json .output n=4}
[
 {
  "data": {
   "image/png": "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\ngaAIPxDUfwA1Ps3HFfyr2AAAAABJRU5ErkJggg==\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f4426005d68>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

Clearly, the dataset is good for classification task, but we want a dataset for detection. Surely we can produce one by randomly distorting and placing the digits at different positions.

```{.python .input  n=5}
# Make objects not too tricky, so we can save training time
data_shape = 64  # we want a reasonable size as input shape
max_ratio = 1.5  # digits not too wide or tall
min_size = 0.4  # digits not too small
def get_batch(batch):
    batch_size = batch.data[0].shape[0]
    data = mx.nd.zeros((batch_size, 1, data_shape, data_shape))
    label = mx.nd.full((batch_size, 3, 5), -1)
    for k in range(batch_size):
        # generate random width/height for the digits
        w = int(round(random.uniform(min_size, 1) * data_shape))
        h = int(round(random.uniform(min_size, 1) * data_shape))
        # regulate the shape
        if float(w) / h > max_ratio:
            w = int(round(h * max_ratio))
        if float(h) / w > max_ratio:
            h = int(round(w * max_ratio))
        # resize the digit
        orig = batch.data[0][k].reshape((28, 28, 1))
        warped = mx.image.imresize(orig, w, h)
        # randomize the new location
        x0 = random.randint(0, data_shape - w)
        y0 = random.randint(0, data_shape - h)
        # copy warped digits to the canvas
        data[k, 0, y0 : y0 + h, x0 : x0 + w] = warped.reshape((1, 1, h, w))
        # the label is the new location and size of the digits, as [id, xmin, ymin, xmax, ymax]
        cid = batch.label[0][k].asscalar()
        xmin = x0 / data_shape
        ymin = y0 / data_shape
        xmax = (x0 + w) / data_shape
        ymax = (y0 + h) / data_shape
        det_label = mx.nd.array([cid, xmin, ymin, xmax, ymax])
        label[k, 0, :] = det_label.reshape((1, 1, 5))
    return mx.io.DataBatch(data=[data], label=[label], index=batch.index, pad=batch.pad)
```

Now, with `get_batch` function, we are getting data for detection task!

```{.python .input  n=6}
det_batch = get_batch(batch)
demo = det_batch.data[0][0].asnumpy()  # grab the first image, convert to numpy array
demo = demo.transpose((1, 2, 0))  # we want channel to be the last dimension
plt.imshow(demo[:, :, (0, 0, 0)])  # convert to 3-channels and display
label = det_batch.label[0][0][0].asnumpy()
print('label for detection:', label)
xmin, ymin, xmax, ymax = [int(x * data_shape) for x in label[1:5]]
rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=(1, 1, 1))
plt.gca().add_patch(rect)
plt.show()
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "label for detection: [ 8.        0.25      0.015625  0.96875   0.5     ]\n"
 },
 {
  "data": {
   "image/png": "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\nO6QnyRQACfQAAAAASUVORK5CYII=\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f4424514828>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

The white bounding box around digit is the desired detection box, namely, ground-truth.

## Detection network

There are multiple convolutional neural network meta-structures specifically designed for object detection. In this section we use Single-Shot Detector (SSD) as an example, and illustrate how a general object detection training/test workflow looks like.

### There are three key points making SSD network different from normal convolutional neural networks:

Details available at https://arxiv.org/abs/1512.02325

* Multi-scale feature maps for detection

* Multiple anchors boxes with different scales and aspect ratios

* Convolutional predictors

### In summary, we need:
* A backbone network producing feature maps in different sizes. We can achieve this by stacking `Conv-BatchNorm-Relu` layers followed by `MaxPooling` layer to reduce the feature map size by a factor of 2. For a 64x64 input image, this could produce feature maps with size 32x32, 16x16, 8x8, 4x4, 2x2, 1x1. 

* Anchor boxes generator. We want multiple anchors with different scales and aspect ratios, so that there always exist an anchor box highly overlaps with an object on image. In this tutorial, we are going to use the following parameters:

    8x8 --> sizes [0.37, 0.447] --> ratios [1, 2, 0.5]

    4x4 --> sizes [0.54, 0.619] --> ratios [1, 2, 0.5]

    2x2 --> sizes [0.71, 0.79]  --> ratios [1, 2, 0.5]

    1x1 --> sizes [0.88, 0.961] --> ratios [1, 2, 0.5]
    
    16x16 feature map might be good for smaller objects, but we skip it in this tutorial in trade of faster convergence.

* 3x3 Convolutional layers are responsible for predict class probabilities, as well as the box deformation predictions. Each convolution channel is responsible for a certain (scale, ratio, feature map) combination.

### Training target in one sentence
![](https://user-images.githubusercontent.com/3307514/28603710-9e7d4d80-717a-11e7-95ed-dd08763fed87.png)

Applying convolutional predictors to feature maps in a sliding window fashion (convolutional built-in characteristics), predicting whether the correspoding anchor is an object (and what class) or background, and how much deformation the corresponding anchor box should transform to cover the object.


```{.python .input  n=7}
class ToySSD(gluon.Block):
    def __init__(self, num_class, **kwargs):
        super(ToySSD, self).__init__(**kwargs)
        # sizes control the scale of anchor boxes, with decreasing feature map size,
        # the anchor boxes are expected to be larger in design
        self.sizes = [[.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        # ratios control the aspect ratio of anchor boxes, here we use 1, 2, 0.5
        self.ratios = [[1,2,.5], [1,2,.5], [1,2,.5], [1,2,.5]]
        num_anchors = [len(x) + len(y) - 1 for x, y in zip(self.sizes, self.ratios)]
        self.num_anchors = num_anchors
        self.num_class = num_class
        with self.name_scope():
            # first build a body as feature
            self.body = nn.HybridSequential()
            # 64 x 64
            # make basic block is a stack of sequential conv layers, followed by
            # a pooling layer to reduce feature map size
            self.body.add(self._make_basic_block(16))
            # 32 x 32
            self.body.add(self._make_basic_block(32))
            # 16 x 16
            self.body.add(self._make_basic_block(64))
            # 8 x 8
            # use cls1 conv layer to get the class predictions on 8x8 feature map
            # use loc1 conv layer to get location offsets on 8x8 feature map
            # use blk1 conv block to reduce the feature map size again
            self.cls1 = nn.Conv2D(num_anchors[0] * (num_class + 1), 3, padding=1)
            self.loc1 = nn.Conv2D(num_anchors[0] * 4, 3, padding=1)
            self.blk1 = self._make_basic_block(64)
            # 4 x 4
            self.cls2 = nn.Conv2D(num_anchors[1] * (num_class + 1), 3, padding=1)
            self.loc2 = nn.Conv2D(num_anchors[1] * 4, 3, padding=1)
            self.blk2 = self._make_basic_block(64)
            # 2 x 2
            self.cls3 = nn.Conv2D(num_anchors[2] * (num_class + 1), 3, padding=1)
            self.loc3 = nn.Conv2D(num_anchors[2] * 4, 3, padding=1)
            # 1 x 1
            self.cls4 = nn.Conv2D(num_anchors[3] * (num_class + 1), 3, padding=1)
            self.loc4 = nn.Conv2D(num_anchors[3] * 4, 3, padding=1)

    def _make_basic_block(self, num_filter):
        """Basic block is a stack of sequential convolution layers, followed by
        a pooling layer to reduce feature map. """
        out = nn.HybridSequential()
        out.add(nn.Conv2D(num_filter, 3, strides=1, padding=1))
        out.add(nn.BatchNorm(in_channels=num_filter))
        out.add(nn.Activation('relu'))
        out.add(nn.Conv2D(num_filter, 3, strides=1, padding=1))
        out.add(nn.BatchNorm(in_channels=num_filter))
        out.add(nn.Activation('relu'))
        out.add(nn.MaxPool2D())
        return out

    def forward(self, x):
        anchors = []
        loc_preds = []
        cls_preds = []
        x = self.body(x)
        # 8 x 8, generate anchors, predict class and location offsets with conv layer
        # transpose, reshape and append to list for further concatenation
        anchors.append(MultiBoxPrior(x, sizes=self.sizes[0], ratios=self.ratios[0]))
        loc_preds.append(nd.flatten(nd.transpose(self.loc1(x), axes=(0, 2, 3, 1))))
        cls_preds.append(nd.flatten(nd.transpose(self.cls1(x), axes=(0, 2, 3, 1))))
        x = self.blk1(x)
        # 4 x 4
        anchors.append(MultiBoxPrior(x, sizes=self.sizes[1], ratios=self.ratios[1]))
        loc_preds.append(nd.flatten(nd.transpose(self.loc2(x), axes=(0, 2, 3, 1))))
        cls_preds.append(nd.flatten(nd.transpose(self.cls2(x), axes=(0, 2, 3, 1))))
        x = self.blk2(x)
        # 2 x 2
        anchors.append(MultiBoxPrior(x, sizes=self.sizes[2], ratios=self.ratios[2]))
        loc_preds.append(nd.flatten(nd.transpose(self.loc3(x), axes=(0, 2, 3, 1))))
        cls_preds.append(nd.flatten(nd.transpose(self.cls3(x), axes=(0, 2, 3, 1))))
        # we use pooling directly here without convolution layers
        x = nd.Pooling(x, global_pool=True, pool_type='max', kernel=(2, 2))
        # 1 x 1
        anchors.append(MultiBoxPrior(x, sizes=self.sizes[3], ratios=self.ratios[3]))
        loc_preds.append(nd.flatten(nd.transpose(self.loc4(x), axes=(0, 2, 3, 1))))
        cls_preds.append(nd.flatten(nd.transpose(self.cls4(x), axes=(0, 2, 3, 1))))
        # concat multiple layers
        anchors = nd.reshape(nd.concat(*anchors, dim=1), shape=(0, -1, 4))
        loc_preds = nd.concat(*loc_preds, dim=1)
        cls_preds = nd.reshape(nd.concat(*cls_preds, dim=1), (0, -1, self.num_class+1))
        cls_preds = nd.transpose(cls_preds, axes=(0, 2, 1))
        return [anchors, cls_preds, loc_preds]
```

ToySSD network takes a batch of images `(64, 1, 64, 64)` as input, and will output three components:

1. Concatenated anchors with shape `(1, 340, 4)` scattered around images with all kinds of scales and aspect ratios. The first dimension is 1, indicating all anchors are shared across batches, second is the number of anchors, the third dimension is for box coordinates (xmin, ymin, xmax, ymax)

2. Concatenated class predictions with shape `(64, 11, 340)`, first dimension is the `batch_size`, second is per-class prediction for `(background, 0, 1, ..., 9)`, the last dimension is number of anchors.

3. Concatenated location predictions with shape `(64, 340 * 4)`.

## Losses

Two types of loss functions are involved here. 
* Cross-entropy loss is typical for softmax classification problems, which serves as classification loss for all anchor boxes. 

* L2Loss is used to penalize incorrect bounding box offsets. We can play with L1Loss, SmoothL1Loss as well.

```{.python .input  n=8}
cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()  # for softmax classification, typical for classification
loc_loss = gluon.loss.L2Loss()  # typical for regression
```

## Evaluate metrics

Evalute metrics are for collectting training status, which has no effect on training itself.

```{.python .input  n=9}
cls_metric = mx.metric.Accuracy()
loc_metric = mx.metric.MAE()
```

## Create network

```{.python .input  n=10}
net = ToySSD(num_class)  # create a net to predict 10-class digits
```

## Set the context

```{.python .input  n=11}
ctx = mx.gpu()  # it takes too long to train using CPU
try:
    _ = nd.zeros(1, ctx=ctx)
except mx.base.MXNetError as err:
    print('No GPU enabled, fall back to CPU, which will be slow...')
    ctx = mx.cpu()
```

## Initalize parameters

```{.python .input  n=12}
net.collect_params().initialize(mx.init.Xavier(magnitude=2.34), ctx=ctx)
```

## Trainer

```{.python .input  n=13}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

## Start training

```{.python .input  n=14}
epochs = 8  # set larger to get better performance
log_interval = 500
for epoch in range(epochs):
    # reset iterator and tick
    train_data.reset()
    cls_metric.reset()
    loc_metric.reset()
    tic = time.time()
    # iterate through all batch
    for i, batch in enumerate(train_data):
        btic = time.time()
        det_batch = get_batch(batch)
        # record gradients
        with ag.record():
            x = det_batch.data[0].as_in_context(ctx)
            y = det_batch.label[0].as_in_context(ctx)
            anchors, cls_preds, loc_preds = net(x)
            z = MultiBoxTarget(*[anchors, y, cls_preds])
            loc_target = z[0]  # loc offset target for (x, y, width, height)
            loc_mask = z[1]  # mask is used to ignore predictions we don't want to penalize
            cls_target = z[2]  # cls_target is an array of labels for all anchors boxes
            # losses
            loss1 = cls_loss(nd.transpose(cls_preds, (0, 2, 1)), cls_target)
            loss2 = loc_loss(loc_preds * loc_mask, loc_target)
            # sum all losses
            loss = loss1 + loss2
            # backpropagate
            loss.backward()
        # apply optimizer
        trainer.step(batch_size)
        # update metrics
        cls_metric.update([cls_target], [cls_preds])
        loc_metric.update([loc_target], [loc_preds * loc_mask])
        if i % log_interval == 0:
            name1, val1 = cls_metric.get()
            name2, val2 = loc_metric.get()
            print('[Epoch %d Batch %d] speed: %f samples/s, training: %s=%f, %s=%f' 
                  %(epoch ,i, batch_size/(time.time()-btic), name1, val1, name2, val2))
    
    # end of epoch logging
    name1, val1 = cls_metric.get()
    name2, val2 = loc_metric.get()
    print('[Epoch %d] training: %s=%f, %s=%f'%(epoch, name1, val1, name2, val2))
    print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
    
# we can save the trained parameters to disk
# net.save_params('ssd_%d.params' % epochs)
```

```{.json .output n=14}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[Epoch 0 Batch 0] speed: 63.873648 samples/s, training: accuracy=0.168796, mae=0.025306\n[Epoch 0 Batch 500] speed: 731.318177 samples/s, training: accuracy=0.978992, mae=0.014439\n[Epoch 0] training: accuracy=0.980648, mae=0.012896\n[Epoch 0] time cost: 82.171110\n[Epoch 1 Batch 0] speed: 761.799972 samples/s, training: accuracy=0.984007, mae=0.009442\n[Epoch 1 Batch 500] speed: 779.345587 samples/s, training: accuracy=0.983687, mae=0.010100\n[Epoch 1] training: accuracy=0.984245, mae=0.009854\n[Epoch 1] time cost: 75.744250\n[Epoch 2 Batch 0] speed: 801.397950 samples/s, training: accuracy=0.986397, mae=0.008735\n[Epoch 2 Batch 500] speed: 801.020109 samples/s, training: accuracy=0.986059, mae=0.009199\n[Epoch 2] training: accuracy=0.986528, mae=0.009079\n[Epoch 2] time cost: 75.629008\n[Epoch 3 Batch 0] speed: 799.493253 samples/s, training: accuracy=0.986994, mae=0.009463\n[Epoch 3 Batch 500] speed: 797.431730 samples/s, training: accuracy=0.987933, mae=0.008785\n[Epoch 3] training: accuracy=0.988277, mae=0.008705\n[Epoch 3] time cost: 75.290620\n[Epoch 4 Batch 0] speed: 807.341676 samples/s, training: accuracy=0.987960, mae=0.009394\n[Epoch 4 Batch 500] speed: 779.984182 samples/s, training: accuracy=0.989378, mae=0.008376\n[Epoch 4] training: accuracy=0.989637, mae=0.008299\n[Epoch 4] time cost: 75.261538\n[Epoch 5 Batch 0] speed: 840.399782 samples/s, training: accuracy=0.989752, mae=0.007622\n[Epoch 5 Batch 500] speed: 790.092350 samples/s, training: accuracy=0.990405, mae=0.008082\n[Epoch 5] training: accuracy=0.990609, mae=0.008001\n[Epoch 5] time cost: 73.699337\n[Epoch 6 Batch 0] speed: 815.184777 samples/s, training: accuracy=0.990395, mae=0.007929\n[Epoch 6 Batch 500] speed: 877.630110 samples/s, training: accuracy=0.991280, mae=0.007763\n[Epoch 6] training: accuracy=0.991397, mae=0.007702\n[Epoch 6] time cost: 71.086407\n[Epoch 7 Batch 0] speed: 820.451910 samples/s, training: accuracy=0.990487, mae=0.008108\n[Epoch 7 Batch 500] speed: 830.539829 samples/s, training: accuracy=0.991831, mae=0.007512\n[Epoch 7] training: accuracy=0.991931, mae=0.007435\n[Epoch 7] time cost: 73.209482\n"
 }
]
```

## Display results

```{.python .input  n=17}
# if pre-trained model is provided, we can load it
# net.load_params('ssd_%d.params' % epochs, ctx)
test_data.reset()
count = 0
limit = 5
pens = dict()
thresh = 0.1
while count < limit:
    x = get_batch(test_data.next()).data[0].as_in_context(ctx)
    # network inference
    anchors, cls_preds, loc_preds = net(x)
    # convert predictions to probabilities
    cls_probs = nd.SoftmaxActivation(cls_preds, mode='channel')
    # apply shifts to anchors boxes, non-maximum-suppression, etc...
    output = MultiBoxDetection(*[cls_probs, loc_preds, anchors], force_suppress=True, clip=True)
    for k, out in enumerate(output.asnumpy()):
        img = x[k].asnumpy().transpose((1, 2, 0))
        img = img[:, :, (0, 0, 0)]
        if count >= limit:
            break
        count += 1
        # display results
        plt.clf()
        plt.imshow(img)
        for det in out:
            cid = int(det[0])
            if cid < 0:
                continue
            score = det[1]
            if score < thresh:
                continue
            if cid not in pens:
                pens[cid] = (random.random(), random.random(), random.random())
            xmin, ymin, xmax, ymax = [int(p * data_shape) for p in det[2:6]]
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, 
                                 edgecolor=pens[cid], linewidth=3)
            plt.gca().add_patch(rect)
            text = class_names[cid]
            plt.gca().text(xmin, ymin-2, '{:s} {:.3f}'.format(text, score),
                           bbox=dict(facecolor=pens[cid], alpha=0.5),
                           fontsize=12, color='white')
            plt.show()
                
```

```{.json .output n=17}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f4424532400>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "data": {
   "image/png": "iVBORw0KGgoAAAANSUhEUgAAAP4AAAD8CAYAAABXXhlaAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAGeRJREFUeJzt3XmcVtWd5/HPj30RBASRfRNEFEGCIXScsI0GMS0mMbFD\nojhhwqtnZNSJ3UbsiZ3YPR3tScekY+JLJsbQaRTE6ICQqEiDKCiyK0tYRJZCFpFNEUqWM388l8u5\nl6eoh6pnqeJ8368Xr/o995xbzyme+tU95y7nmHMOEQlLnVI3QESKT4kvEiAlvkiAlPgiAVLiiwRI\niS8SICW+SICqlfhmNtLM1pvZJjO7P1+NEpHCsqrewGNmdYENwHVAGbAE+JZzbm3+micihVCvGvt+\nHtjknNsMYGZTgdFAhYlvZrpNUKTAnHNWWZ3qdPU7ANu912XRNhGp4apzxM+JmY0Hxhf6fUQkd9VJ\n/B1AJ+91x2hbgnNuEjAJ1NUXqSmq09VfAvQ0s25m1gD4K2BmfpolIoVU5SO+c+64mU0AXgbqAr91\nzq3JV8PqNGlE3YYN8/XtaowT5eWc/PRoqZshgavy5bwqvdk5dPXrt7yQVoMGFLI5JbFv8XKO7T9Y\n6mbIeazQZ/VFpJZS4osESIkvEqBalfiP/e1EVk2ZzsY/vMjC/zuZMV8eddb6Xx06nKW/e4bNL8zm\nqR8+RIsLmlVY97pBg5n/+JO89/xsXvyXX9Krc5es9ab/5Kfs+tN/ULdO8r/uv47+Gm8/NYXNL8xm\nwRNP0b1Dx3P/AUWKpFYl/i+nPcM1d4yh59f/krE//l/cf/t3uerSnlnrXta5K//nf3yfCT/9CVd+\n6+scKS/n4Ql3Z63brX0Hfn3fA9z32KP0uuUveWXxIib//T+ekdxfGzaCenXPvBAy5sujGPPlG/jO\n3z9A96/eyG0/+jv2HdQJPKm5alXir9+2hc+OHQPAOXA4urZrn7Xu14aN4JXFb/LW6nf49OhRHvn9\nU4z6i/9E08aNz6g77HPXsHj1u7y9ZjUnTp7kselTueSi1gy+ql9cp1mTptw75nb+4bdPJPY1M+79\n9u08OOnXbNi2FYCtOz/gwCcf5+vHFsm7WpX4AA/feTebX/gjC38zmd379vHqksVZ613WpStr338v\nfr115wccO36cHhV0wc0sEZsZvbt0i7c9cMc4Js9+kQ/37Uvs1751Gzq0uZjeXbqx7N+m8vZTU/jb\n74xNfD+RmqbWJf79v/oFl379K9z0N3fxx4Wvxz2AtKaNG3Po8OHEtkOHD3NB4yZn1F2wYhmD+17F\nX/TtR/169bj71jE0qFePxtENRP169uKaPlfy5Mznz9i3Xes2AAwdMJBh/20cX//B97l5yPBKzz+I\nlFKtS3yAkydP8vaa1bRr3ZqxN96Utc7hI0do1iSZ5M2aNOGTI5+eUXdT2Xbu+pdH+Kf/fherpkyn\nVfML2bBtKzv37sXMePjOe/jhE49x4uTJM/Y9+lk5AL96biqHDh9m+57d/P5PsxhxzaA8/KQihVHw\np/MKqV7duhWO8ddv3UKf7j3i150vaUeD+vV5b0dZ1vqz3ljArDcWANC8aVPGfHkUKzf8mWZNmtCv\nZy+euP9BAOrWzfytXPH7Z/neP/2YdzZtoPzYZ/g3QGp1Iqnpas0Rv/WFLRg9ZBhNGjWiTp06DB0w\nkK8OHc7rK1dkrf/8vLlcP2gwg67oS5OGjbjvtjv446LXOXzkSNb6V13akzp16nDRhRfy07vu5eW3\nFrGpbDuHDh+m33e+wYgJ32PEhO/x7QcnAnD9XX/N8vXrOFJezszX5nPnLbfStHFj2rVuzXdu+Apz\nFr9ZsP8LkeqqNUd8h+OOG2/inyf8T+rUMcp27+aHT/yaVxYvylp//bYt3PfLR/nVfQ/QqnlzFqxY\nzj0/++e4/OmHfsJba97lX6c9DcA//PUErujWg2MnjvPi66/xo0mPx3U/3L8/jhvVbxBt2xd3/Sc+\n/q/89K7vs+rfp3Pw8CdMeWk2z7zyp7z/H4jkix7SKTI9pCOFpod0RCQrJb5IgJT4IgFS4osESIkv\nEqBacTnvkpHDSt2E/HGOS24YXupW1Hir7nmw1E04r9XYxD9RXs6+xcszL86jO+HqNm5U6iaI1NzE\nP/np0Xg2Wh0hRfKrxiZ+RdQFPH/1+/lDpW5CMHRyTyRASnyRACnxRQJUaeKb2W/NbI+Zrfa2tTKz\nOWa2MfrasrDNFJF8yuWI/ztgZGrb/cBc51xPYG70WkRqiUoT3zm3ANiX2jwamBzFk4Gb89wuESmg\nqo7x2zrndkbxLqBtntojIkVQ7ev4zjl3tgk2zGw8ML667yMi+VPVI/5uM2sHEH3dU1FF59wk59xA\n59zAKr6XiORZVRN/JjA2iscCM/LTHBEphlwu5z0DvAlcZmZlZjYOeBi4zsw2Av85ei0itUSlY3zn\n3LcqKBqR57aISJHozj2RACnxRQKkxBcJkBJfJEBKfJEAKfFFAqTEFwmQEl8kQEp8kQAp8UUCpMQX\nCZASXyRASnyRACnxRQKkxBcJkBJfJEBKfJEAKfFFAqTEFwmQEl8kQEp8kQAp8UUCpMQXCZASXyRA\nSnyRAOWyhFYnM5tnZmvNbI2Z3R1tb2Vmc8xsY/S1ZeGbKyL5kMsR/zhwr3OuD/AF4E4z6wPcD8x1\nzvUE5kavRaQWyGXtvJ3Azij+2MzWAR2A0cDQqNpkYD7wg4K08jxlZonXdeqc/jt84sSJCvdr2fJ0\n56pt27aJsgsuuCDrPuntLVq0iOP69esnypYsWRLHe/fujeMjR44k6p2tjVKzndMY38y6AlcDi4G2\n0R8FgF1A2wp2E5EaptIj/ilmdgHwB+Ae59wh/2jlnHNm5irYbzwwvroNFZH8yemIb2b1yST9FOfc\n89Hm3WbWLipvB+zJtq9zbpJzbqBzbmA+Giwi1VfpEd8yh/YngXXOuZ95RTOBscDD0dcZBWlhCaXH\n4M5l7dRU+Xs2aNAgUdaoUaM4/vjjjyv8Hp07d47jIUOGJMo6deqU9b3at2+fqNe7d+84btWqVaLs\ngQceiONly5bF8QcffJCod7Y2Ss2WS1f/i8BtwLtmtjLa9gCZhH/WzMYBW4FvFqaJIpJvuZzVfwOw\nCopH5Lc5IlIMOZ/cO5/4XeCGDRsmypo2bZo1Bjhw4EAcHz58OI7PdlnLv2wGcPHFF8dx3759E2X+\n6yeeeCKOP/3000S9AQMGxPHNN9+cKCsrK4tj/7Lfzp07E/XWrVsXx8OHD0+UTZgwIWu9F198MVFv\n1qxZcXz8+HGk9tAtuyIBUuKLBCjIrn69eqd/7PQdbf4Z8yuuuCJRtmvXrjjesWNHHH/yySeJes2b\nN4/jyy+/PFHWrVu3Cr9/x44d43jKlClxXLdu3UQ9//WePcmrqJs2bYpj/yqBvx2S/wfp4ciXvvSl\nOPavEvhDHYC1a9fG8YYNG5DaQ0d8kQAp8UUCpMQXCVDwY3z/khckx+S33XZbosy/JLZq1ao49sf+\nkLwrbtSoUYmyHj16xHF67L5+/fo49p/US4/B/bH1ypUrE2UfffRR1nYdPXo0Uc+/Wy/d/quvvjqO\nL7nkkjgeNGhQhe3QGL920RFfJEBKfJEABdnV/+yzz+LY7/YDdO/ePY47dOiQKOvSpUsc+0OC9AQV\n/h1/6QdbFixYEMfbt29PlPld7t27d1fYfr/NJ0+eTJSVl5dnjdMPGPl3L6Yn4qionv/zAwwePDiO\np02blijz7zZMt1FKT0d8kQAp8UUCpMQXCVCQY3z/abr0+NN/Ws+/5RWSE2f44/r0hBT++HbRokWJ\nsoULF8bx+++/X+F+hw4dqrCN+XDRRRfFsX/5Ds58YjHbPpC8NJnex798qDF+zaMjvkiAlPgiAQqy\nq+9LX4rzn7rzu9uQvHPvpZdeiuM333wzUc9/ws/v2kPyzrr0BB7+Jbd8zO93Nr169Yrj8eOTkyBX\nNDd/+rJf48aN49i/01BqPn1aIgFS4osEKPiuvt/1Bpg/f34cpye58OfZ27p1axyn57Pz78Dbt29f\nouzYsWNVbuu58if2+NznPpcoGz16dBynHwLyrVmzJo79pbUA5s2bF8fpKxtaXqtm0xFfJEBKfJEA\nKfFFAhT8GN8ft0NyHnk/PhfpSSnzzZ/Aw7+kBsm76/yn50aOHJmod+2118ax/7QiJM9R+Oc80vPq\nv/XWW3GcnuhDarZKj/hm1sjM3jazVWa2xsx+HG3vZmaLzWyTmU0zswaVfS8RqRly6eqXA8Odc/2A\n/sBIM/sC8AjwqHPuUmA/MK5wzRSRfMpl7TwHnJo4vn70zwHDgTHR9snAj4DH899ESfMfHvInDgG4\n/vrr49i/ZOffqQfJh2rSlyNnzpwZx1OnTo3j1atXJ+qpe1975XRyz8zqRivl7gHmAO8BB5xzpxZM\nKwM6VLS/iNQsOSW+c+6Ec64/0BH4PNC7kl1iZjbezJaa2dIqtlFE8uycLuc55w4A84DBQAszOzVU\n6AjsqGCfSc65gc65gdVqqYjkTaVjfDNrAxxzzh0ws8bAdWRO7M0DbgGmAmOBGYVs6PkoPdGn/1Sc\nf1muXbt2iXr+RJ/+ktmQXGrbnyw0vb6fv5y2/0QiwN69e+PYvzSZfpJRaq9cruO3AyabWV0yPYRn\nnXOzzGwtMNXM/hFYATxZwHaKSB7lclb/HeDqLNs3kxnvi0gtE/yde1XlXw5r0qRJosyfiz69TJY/\n5376qbg2bdrEcdeuXeM4fSnOf9LO79pDcrjw3nvvxXH6LkS/O59uo383o56yOz/pXn2RACnxRQKk\nrn4V+Wfa/bPskJybrnnz5omy/v37V7if3/X3V6lNDyX8B3PSVwb2798fx3Pnzo3jWbNmJer5Q47b\nb789UXbw4ME49pfhkvOHjvgiAVLiiwRIiS8SII3xq8i/xHbrrbcmypo1axbH6aWl/Akw/XE8JJea\nWr58eRx36tQpUc+/qy89Iej06dPjePHixXGcfrLOb+Ps2bMTZStWrIjj9NoCcn7QEV8kQEp8kQCp\nq19FXbp0ieNRo0Ylyvyu+LksheXPwf/qq6/G8VVXXZWod9lll8Xx5s2bE2WPPfZYHPur76bvwPMv\n+z333HOJMr97X+ilvKQ0dMQXCZASXyRASnyRAGmMX0X+pbf0enj+03lpZxsz+/v58ZYtWxL1GjQ4\nPZN5es26s43rfcePH4/j9CQdGtef/3TEFwmQEl8kQOrqV9HGjRvjeM6cOYmyG264IY7TS1z5d/Kl\nJ8Dw58sfMmRIHPuX9iB5p53ftYfcJ87wu/OabCM8OuKLBEiJLxIgdfWr6J133onjp59+OlHmT9KR\nfhDHn2zDnx8PksOCG2+8MY79FWsBXn/99Tj2584TyZWO+CIBUuKLBEiJLxIgjfGryL/bbdmyZYmy\niRMnxrH/FB/AxRdfHMe9eyfXHh0+fHgc9+zZM45vueWWRD1/As/0pb5FixZV2naRnI/40VLZK8xs\nVvS6m5ktNrNNZjbNzBpU9j1EpGY4l67+3YC/HMsjwKPOuUuB/cC4fDZMRAonp66+mXUEbgT+N/B9\nyzxBMhwYE1WZDPwIeLwAbayR/IdcPvroo0SZ/3rbtm2JMr+b7t/9B8mlq7773e/GsT/xBkCdOqf/\nXvtz+ENyMg//4Z6jR4+e+UNIsHI94v8cuA849UjaRcAB59yp3/4yoEO2HUWk5qk08c3sK8Ae59yy\nyupWsP94M1tqZkursr+I5F8uXf0vAjeZ2SigEdAc+AXQwszqRUf9jsCObDs75yYBkwDMTA96i9QA\nlSa+c24iMBHAzIYCf+Oc+7aZTQduAaYCY4EZBWxnrZW+pdZ/vX379kTZjh2n/3aOHDkyjtOX/fxl\nstNLbe/atSuOZ8w4/ZF88MEHiXr+RCISnurcwPMDMif6NpEZ8z+ZnyaJSKGd0w08zrn5wPwo3gx8\nPv9NEpFC0517JeRfEoTkpTj/ibyWLVsm6nXu3DmO/ScBAe6444449ifpSN/hlx5mSFh0r75IgJT4\nIgEKvqufngq7Xr3T/yXpOfF8/jx16em1c5WextqfKtufxy89YcewYcPiuHv37omyvn37xvGAAQPi\n+N13303UU1c/bDriiwRIiS8SICW+SICCH+P7Y3pIjqf9ee4heT7Av1SWXsaqqvPU+0/Qvfzyy3H8\n4YcfJur5T+f16NEjUeb/PP7431+6W0RHfJEAKfFFAhRkV9/vsvfq1StR9o1vfCOOL7/88kSZv0pt\nWVlZHC9ZsiRR77XXXovjAwcOJMr8IUKulwEPHTqUeJ1e3bYiWvVWKqIjvkiAlPgiAVLiiwQoyDG+\nP0Flet77m266KY7TE2D4Y3x/YotOnTol6vlP0/kTXkLy0lx5eXmF39+X/v7pJ/Iq4k/6ket5AQmD\njvgiAVLiiwQoyK6+36VOL2N9xRVXxLF/h1yav58/Px7AddddF8crVqxIlG3evDmO05fzWrdunfW9\n2rdvn3jdsWPHCtvlW7t2bRz7c/GJ6IgvEiAlvkiAguzqf/bZZ3G8evXqRNnjj59eBezKK69MlDVs\n2DCO27ZtG8f+HHiQfLjHX/U2vV96iuv0Q0GnNG3aNPHavyrx/vvvJ8pmz54dxwsXLozj9DTfEjYd\n8UUCpMQXCZASXyRAQY7x/fnsN23alCibOnVqHF9zzTWJMn8M7o/r+/Tpk6jnT3rhL4sN0KZNmzhO\nT+bpL6998ODBrNsh+bSef8ku3f7169fHcXqyEAlbTolvZluAj4ETwHHn3EAzawVMA7oCW4BvOuf2\nF6aZIpJP59LVH+ac6++cGxi9vh+Y65zrCcyNXotILVCdrv5oYGgUTyazpt4PqtmeovAvo6Unyli+\nfHkcpy/1+RN4+JfY0nfS9evXL479lW0BhgwZEseNGzdOlM2bNy+Oly5dGsfpbrq/qm56qLJ79+44\n9u8M1KQc4sv1iO+AV8xsmZmNj7a1dc7tjOJdQNvsu4pITZPrEf9a59wOM7sYmGNmf/YLnXPOzLIe\nUqI/FOOzlYlIaeR0xHfO7Yi+7gFeILM89m4zawcQfd1Twb6TnHMDvXMDIlJilR7xzawpUMc593EU\nXw88BMwExgIPR19nFLKhhZK+bda/ndeP044cOZI1huSluA0bNiTK3njjjThOz+m/bdu2OPafpktP\n2OFPquG/V2VtFjkll65+W+CF6MRWPeBp59xLZrYEeNbMxgFbgW8Wrpkikk+VJr5zbjPQL8v2j4AR\nhWiUiBRWkHfu5YN/91+6u+2/3rhxY9HaJJIr3asvEiAlvkiAlPgiAVLiiwRIiS8SICW+SICU+CIB\nqnXX8fv9/KFSN0Gk1tMRXyRASnyRAFkxZ2ap6Jl9Eckf55xVVkdHfJEAKfFFAqTEFwmQEl8kQEp8\nkQAp8UUCpMQXCZASXyRASnyRACnxRQKkxBcJkBJfJEBKfJEA5ZT4ZtbCzJ4zsz+b2TozG2xmrcxs\njpltjL62LHRjRSQ/cj3i/wJ4yTnXm8xyWuuA+4G5zrmewNzotYjUApU+j29mFwIrge7Oq2xm64Gh\nzrmd0TLZ851zl1XyvfQ8vkiB5et5/G7Ah8BTZrbCzH4TLZfd1jm3M6qzi8yquiJSC+SS+PWAAcDj\nzrmrgcOkuvVRTyDr0dzMxpvZUjNbWt3Gikh+5JL4ZUCZc25x9Po5Mn8IdkddfKKve7Lt7Jyb5Jwb\n6JwbmI8Gi0j1VZr4zrldwHYzOzV+HwGsBWYCY6NtY4EZBWmhiORdTpNtmll/4DdAA2Az8F/I/NF4\nFugMbAW+6ZzbV8n30ck9kQLL5eSeZtkVOc9oll0RyUqJLxIgJb5IgJT4IgFS4osESIkvEiAlvkiA\n6hX5/faSudmndRSXUk1oA6gdaWpH0rm2o0sulYp6A0/8pmZLS33vfk1og9qhdpSqHerqiwRIiS8S\noFIl/qQSva+vJrQB1I40tSOpIO0oyRhfREpLXX2RABU18c1spJmtN7NNZla0WXnN7LdmtsfMVnvb\nij49uJl1MrN5ZrbWzNaY2d2laIuZNTKzt81sVdSOH0fbu5nZ4ujzmWZmDQrZDq89daP5HGeVqh1m\ntsXM3jWzlaemiSvR70hRprIvWuKbWV3gV8ANQB/gW2bWp0hv/ztgZGpbKaYHPw7c65zrA3wBuDP6\nPyh2W8qB4c65fkB/YKSZfQF4BHjUOXcpsB8YV+B2nHI3mSnbTylVO4Y55/p7l89K8TtSnKnsnXNF\n+QcMBl72Xk8EJhbx/bsCq73X64F2UdwOWF+stnhtmAFcV8q2AE2A5cAgMjeK1Mv2eRXw/TtGv8zD\ngVmAlagdW4DWqW1F/VyAC4H3ic69FbIdxezqdwC2e6/Lom2lUtLpwc2sK3A1sLgUbYm61yvJTJI6\nB3gPOOCcOx5VKdbn83PgPuBk9PqiErXDAa+Y2TIzGx9tK/bnUrSp7HVyj7NPD14IZnYB8AfgHufc\noVK0xTl3wjnXn8wR9/NA70K/Z5qZfQXY45xbVuz3zuJa59wAMkPRO83sS35hkT6Xak1lfy6Kmfg7\ngE7e647RtlLJaXrwfDOz+mSSfopz7vlStgXAOXcAmEemS93CzE49v1GMz+eLwE1mtgWYSqa7/4sS\ntAPn3I7o6x7gBTJ/DIv9uVRrKvtzUczEXwL0jM7YNgD+iswU3aVS9OnBzcyAJ4F1zrmflaotZtbG\nzFpEcWMy5xnWkfkDcEux2uGcm+ic6+ic60rm9+E/nHPfLnY7zKypmTU7FQPXA6sp8ufiijmVfaFP\nmqROUowCNpAZT/5dEd/3GWAncIzMX9VxZMaSc4GNwKtAqyK041oy3bR3yKxHuDL6PylqW4CrgBVR\nO1YDD0bbuwNvA5uA6UDDIn5GQ4FZpWhH9H6ron9rTv1uluh3pD+wNPps/h/QshDt0J17IgHSyT2R\nACnxRQKkxBcJkBJfJEBKfJEAKfFFAqTEFwmQEl8kQP8fGKYM9BHqIeIAAAAASUVORK5CYII=\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f44240d3908>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f44241e1358>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "data": {
   "image/png": "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\npmXNsgAAAABJRU5ErkJggg==\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f442406b4e0>"
  },
  "metadata": {},
  "output_type": "display_data"
 },
 {
  "data": {
   "image/png": "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\ngg==\n",
   "text/plain": "<matplotlib.figure.Figure at 0x7f44243337f0>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

## Conclusion

Detection is harder than classification, since we want not only class probabilities, but also localizations of different objects including potential small objects. Using sliding window together with a good classifier might be an option, however, we have shown that with a properly designed convolutional neural network, we can do single shot detection which is blazing fast and accurate! 

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)

```{.python .input}

```
