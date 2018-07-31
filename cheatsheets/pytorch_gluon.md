# PyTorch to MXNet
This cheatsheet serves as a quick reference for PyTorch users.

## Pytorch Tensor and MXNet NDArray
###  Tensor operation
We document PyTorch function names that are different than MXNet NDArray

| Function                      | PyTorch                                   | MXNet Gluon                                               |
|-------------------------------|-------------------------------------------|-----------------------------------------------------------|
| Element-wise inverse cosine   | `x.acos()` or `torch.acos(x)`             | `nd.arccos(x)`                                            |
| Batch Matrix product and accumulation| `torch.addbmm(M, batch1, batch2)`  | `nd.linalg_gemm(M, batch1, batch2)` Leading n-2 dim are reduced |
| Element-wise division of t1, t2, multiply v, and add t | `torch.addcdiv(t, v, t1, t2)` | `t + v*(t1/t2)`                              |
| Matrix product and accumulation| `torch.addmm(M, mat1, mat2)`             | `nd.linalg_gemm(M, mat1, mat2)`                           |
| Outer-product of two vector add a matrix | `m.addr(vec1, vec2)`           | Not available                                             |
| Element-wise applies function | `x.apply_(calllable)`                     | Not available, but there is `nd.custom(x, 'op')`          |
| Element-wise inverse sine     | `x.asin()` or `torch.asin(x)`             | `nd.arcsin(x)`                                            |
| Element-wise inverse tangent  | `x.atan()` or `torch.atan(x)`             | `nd.arctan(x)`                                            |
| Tangent of two tensor         | `x.atan2(y)` or `torch.atan2(x, y)`       | Not available                                             |
| batch matrix product          | `x.bmm(y)` or `torch.bmm(x, x)`           | `nd.linalg_gemm2(x, y)`                                   |
| Draws a sample from bernoulli distribution | `x.bernoulli()`              | Not available                                             |
| Fills a tensor with number drawn from Cauchy distribution | `x.cauchy_()` | Not available                                             |
| Splits a tensor in a given dim| `x.chunk(num_of_chunk)`                   | `nd.split(x, num_outputs=num_of_chunk)`                   |
| Limits the values of a tensor to between min and max | `x.clamp(min, max)`| `nd.clip(x, min, max)`                                    |
| Returns a copy of the tensor  | `x.clone()`                               | `x.copy()`                                                |
| Cross product                 | `x.cross(y)`                              | Not available                                             |
| Cumulative product along an axis| `x.cumprod(1)`                          | Not available                                             |
| Cumulative sum along an axis  | `x.cumsum(1)`                             | Not available                                             |
| Address of the first element  | `x.data_ptr()`                            | Not available                                             |
| Creates a diagonal tensor     | `x.diag()`                                | Not available                                             |
| Computes norm of a tensor     | `x.dist()`                                | `nd.norm(x)` Only calculate L2 norm                       |
| Computes Gauss error function | `x.erf()`                                 | Not available                                             |
| Broadcasts/Expands tensor to new shape | `x.expand(3,4)`                  | `x.broadcast_to([3, 4])`                                  |
| Fills a tensor with samples drawn from exponential distribution | `x.exponential_()` | `nd.random_exponential()`                      |
| Element-wise mod              | `x.fmod(3)`                               | `nd.module(x, 3)`                                         |
| Fractional portion of a tensor| `x.frac()`                                | `x - nd.trunc(x)`                                         |
| Gathers values along an axis specified by dim | `torch.gather(x, 1,  torch.LongTensor([[0,0],[1,0]]))` | `nd.gather_nd(x, nd.array([[[0,0],[1,1]],[[0,0],[1,0]]]))`  |
| Solves least square & least norm | `B.gels(A)`                            | Not available                                             |
| Draws from geometirc distribution | `x.geometric_(p)`                     | Not available                                             |
| Device context of a tensor    | `print(x)` will print which device x is on| `x.context`                                               |
| Repeats tensor                | `x.repeat(4,2)`                           | `x.tile(4,2)`                                             |
| Data type of a tensor         | `x.type()`                                | `x.dtype`                                                 |
| Scatter                       | `torch.zeros(2, 4).scatter_(1, torch.LongTensor([[2], [3]]), 1.23)` | `nd.scatter_nd(nd.array([1.23,1.23]), nd.array([[0,1],[2,3]]), (2,4))` |
| Returns the shape of a tensor | `x.size()`                                | `x.shape`                                                 |
| Number of elements in a tensor| `x.numel()`                               | `x.size`                                                  |
| Returns this tensor as a NumPy ndarray | `x.numpy()`                      | `x.asnumpy()`                                             |
| Eigendecomposition for symmetric matrix | `e, v = a.symeig()`             | `v, e = nd.linalg.syevd(a)`                               |
| Transpose                     | `x.t()`                                   | `x.T`                                                     |
| Sample uniformly              | `torch.uniform_()`                        | `nd.sample_uniform()`                                     |
| Inserts a new dimesion        | `x.unsqueeze()`                           | `nd.expand_dims(x)`                                       |
| Reshape                       | `x.view(16)`                              | `x.reshape((16,))`                                          |
| Veiw as a specified tensor    | `x.view_as(y)`                            | `x.reshape_like(y)`                                       |
| Returns a copy of the tensor after casting to a specified type | `x.type(type)` | `x.astype(dtype)`                                   |
| Copies the value of one tensor to another | `dst.copy_(src)`              | `src.copyto(dst)`                                         |
| Returns a zero tensor with specified shape | `x = torch.zeros(2,3)`       | `x = nd.zeros((2,3))`                                     |
| Returns a one tensor with specified shape | `x = torch.ones(2,3)`         | `x = nd.ones((2,3)`                                       |
| Returns a Tensor filled with the scalar value 1, with the same size as input | `y = torch.ones_like(x)` | `y = nd.ones_like(x)`       |

###  Functional
###  GPU
Just like Tensor, MXNet NDArray can be copied to and operated on GPU. This is done by specifying
context.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| Copy to GPU            | `y = torch.FloatTensor(1).cuda()` | `y = mx.nd.ones((1,), ctx=mx.gpu(0))`                                      |
| Convert to numpy array | `x = y.cpu().numpy()`             | `x = y.asnumpy()`                                                          |
| Context scope          | `with torch.cuda.device(1):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`y= torch.cuda.FloatTensor(1)`                    | `with mx.gpu(1):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`y = mx.nd.ones((3,5))`      |
###  Cross-device
Just like Tensor, MXNet NDArray can be copied across multiple GPUs.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| Copy from GPU 0 to GPU 1           | `x = torch.cuda.FloatTensor(1)`<br/>`y=x.cuda(1)`| `x = mx.nd.ones((1,), ctx=mx.gpu(0))`<br/>`y=x.as_in_context(mx.gpu(1))`                                      |
| Copy Tensor/NDArray on different GPUs | `y.copy_(x)`             | `x.copyto(y)`                                                          |

## Autograd
### variable wrapper vs autograd scope

Autograd package of PyTorch/MXNet enables automatic differentiation of Tensor/NDArray.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| Recording computation       | `x = Variable(torch.FloatTensor(1), requires_grad=True)`<br/>`y = x * 2`<br/>`y.backward()`  | `x = mx.nd.ones((1,))`<br/>`x.attach_grad()`<br/>`with mx.autograd.record():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`y = x * 2`<br/>`y.backward()`                                   |

### scope override (pause, train_mode, predict_mode)

Some operators (Dropout, BatchNorm, etc) behave differently in training 
and making predictions. This can be controlled with train_mode and predict_mode scope in MXNet.
Pause scope is for codes that do not need gradients to be calculated.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| Scope override   | Not available | `x = mx.nd.ones((1,))`<br/>`with autograd.train_mode():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`y = mx.nd.Dropout(x)`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`with autograd.predict_mode():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`z = mx.nd.Dropout(y)`<br/><br/>`w = mx.nd.ones((1,))`<br/>`w.attach_grad()`<br/>`with autograd.record():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`y = x * w`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`y.backward()`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`with autograd.pause():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`w += w.grad`   |

### batch-end synchronization is needed

MXNet uses lazy evaluation to achieve superior performance. The Python thread just pushes the operations into the backend engine and then returns. In training phase batch-end synchronization is needed, e.g, `asnumpy()`, `wait_to_read()`, `metric.update(...)`.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| Batch-end synchronization    |  Not available  | `for (data, label) in train_data:`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`with autograd.record():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`output = net(data)`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`L = loss(output, label)`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`L.backward()`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`trainer.step(data.shape[0])`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`metric.update([label], [output])` |


## Pytorch module and Gluon blocks
### for new block definition, gluon needs name_scope

name_scope coerces gluon to give each parameter an appropriate name, indicating which model it belongs to.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| New block definition   | `class Net(torch.nn.Module):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`def __init__(self, D_in, D_out):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`super(Net, self).__init__()`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`self.linear = torch.nn.Linear(D_in, D_out)`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`def forward(self, x):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`return self.linear(x)`       |    `class Net(mx.gluon.Block):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`def __init__(self, D_in, D_out):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`super(Net, self).__init__()`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`with self.name_scope():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`self.dense=mx.gluon.nn.Dense(D_out, in_units=D_in)`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`def forward(self, x):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`return self.dense(x)`      |

### Parameter and Initializer

when creating new layers in pytorch, you do not need to specify its parameter initializer, and different layers have different default initializer. When you create new layers in gluon, you can specify its initializer or just leave it none. The parameters will finish initializing after calling `net.initialize(<init method>)` and all parameters will be initialized in `init method` except those layers whose initializer specified.
 

| Function       | PyTorch           | MXNet Gluon        |   
|----------------|-------------------|--------------------|
| Get all parameters |  `net.parameters()` | `net.collect_params()` | 
| Initialize network |  Not Available | `net.initialize(mx.init.Xavier())` |
| Specify layer initializer | `layer = torch.nn.Linear(20, 10)`<br/> `torch.nn.init.normal(layer.weight, 0, 0.01)` | `layer = mx.gluon.nn.Dense(10, weight_initializer=mx.init.Normal(0.01))` |

### usage of existing blocks look alike

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| Usage of existing blocks    |  `y=net(x)`  |  `y=net(x)`   |

### HybridBlock can be hybridized, and allows partial-shape info

HybridBlock supports forwarding with both Symbol and NDArray. After hybridized, HybridBlock will create a symbolic graph representing the forward computation and cache it. Most of the built-in blocks (Dense, Conv2D, MaxPool2D, BatchNorm, etc.) are HybridBlocks.

Instead of explicitly declaring the number of inputs to a layer, we can simply state the number of outputs. The shape will be inferred on the fly once the network is provided with some input.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| partial-shape  <br/> hybridized    |  Not Available   |  `net = mx.gluon.nn.HybridSequential()`<br/>`with net.name_scope():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`net.add(mx.gluon.nn.Dense(10))`<br/>`net.hybridize()`   |

### SymbolBlock

SymbolBlock can construct block from symbol. This is useful for using pre-trained models as feature extractors.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
|  SymbolBlock    |  Not Available   |  `alexnet = mx.gluon.model_zoo.vision.alexnet(pretrained=True, prefix='model_')`<br/>`out = alexnet(inputs)`<br/>`internals = out.get_internals()`<br/>`outputs = [internals['model_dense0_relu_fwd_output']]`<br/>`feat_model = gluon.SymbolBlock(outputs, inputs, params=alexnet.collect_params())`   |


## Pytorch optimizer vs Gluon Trainer
### for gluon zero_grad is not necessary most of the time
`zero_grad` in optimizer(Pytorch) or Trainer(Gluon) clears the gradients of all parameters. In gluon, there is no need to clear the gradients every batch if `grad_req = 'write'`(default).

| Function               | Pytorch                           | MXNet Gluon                              |
|------------------------|-----------------------------------|------------------------------------------|
| clear the gradients |   `optm = torch.optim.SGD(model.parameters(), lr=0.1)`<br/>`optm.zero_grad()`<br/>`loss_fn(model(input), target).backward()`<br/>`optm.step()`    | `trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})`<br/>`with autograd.record():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`loss = loss_fn(net(data), label)`<br/>`loss.backward()`<br/>`trainer.step(batch_size)`      |

### Multi-GPU training

| Function               | Pytorch                           | MXNet Gluon                              |
|------------------------|-----------------------------------|------------------------------------------|
| data parallelism |   `net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])`<br/>`output = net(data)`    | `ctx = [mx.gpu(i) for i in range(3)]`<br/>`data = gluon.utils.split_and_load(data, ctx)`<br/>`label = gluon.utils.split_and_load(label, ctx)`<br/>`with autograd.record():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`losses = [loss(net(X), Y) for X, Y in zip(data, label)]`<br/>`for l in losses:`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`l.backward()`      |

### Distributed training

| Function               | Pytorch                           | MXNet Gluon                              |
|------------------------|-----------------------------------|------------------------------------------|
| distributed data parallelism |   `torch.distributed.init_process_group(...)`<br/>`model = torch.nn.parallel.distributedDataParallel(model, ...)`    | `store = kv.create('dist')`<br/>`trainer = gluon.Trainer(net.collect_params(), ..., kvstore=store)`  |

## Monitoring
### MXNet has pre-defined metrics

Gluon provide several predefined metrics which can online evaluate the performance of a learned model.

| Function               | Pytorch                           | MXNet Gluon                              |
|------------------------|-----------------------------------|------------------------------------------|
| metric |  Not available   | `metric = mx.metric.Accuracy()`<br/>`with autograd.record():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`output = net(data)`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`L = loss(ouput, label)`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`loss(ouput, label).backward()`<br/>`trainer.step(batch_size)`<br/>`metric.update(label, output)`  |

### Data visualization

tensorboardX(PyTorch) and dmlc-tensorboard(Gluon) can be used to visualize your network and plot quantitative metrics about the execution of your graph.

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| visualization    |  `writer = tensorboardX.SummaryWriter()`<br/>`...`<br/>`for name, param in model.named_parameters():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`grad = param.clone().cpu().data.numpy()`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`writer.add_histogram(name, grad, n_iter)`<br/>`...`<br/>`writer.close()` |  `summary_writer = tensorboard.FileWriter('./logs/')`<br/>`...`<br/>`for name, param in net.collect_params():`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`grad = param.grad.asnumpy().flatten()`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`s = tensorboard.summary.histogram(name, grad)`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`summary_writer.add_summary(s)`<br/>`...`<br/>`tensorboard.summary_writer.close()`    |

## I/O and deploy
### Data loading
`Dataset` and `DataLoader` are the basic components for loading data.

| Class               | Pytorch                           | MXNet Gluon                              |
|------------------------|-----------------------------------|------------------------------------------|
| Dataset holding arrays | `torch.utils.data.TensorDataset(data_tensor, label_tensor)`| `gluon.data.ArrayDataset(data_array, label_array)`                        |
| Data loader | `torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, drop_last=False)` | `gluon.data.DataLoader(dataset, batch_size=None, shuffle=False, sampler=None, last_batch='keep', batch_sampler=None, batchify_fn=None, num_workers=0)`|
| Sequentially applied sampler | `torch.utils.data.sampler.SequentialSampler(data_source)` | `gluon.data.SequentialSampler(length)` |
| Random order sampler | `torch.utils.data.sampler.RandomSampler(data_source)` | `gluon.data.RandomSampler(length)`|

Some commonly used datasets for computer vision are provided in `mx.gluon.data.vision` package.

| Class               | Pytorch                           | MXNet Gluon                              |
|------------------------|-----------------------------------|------------------------------------------|
| MNIST handwritten digits dataset. | `torchvision.datasets.MNIST`| `mx.gluon.data.vision.MNIST` |
| CIFAR10 Dataset. | `torchvision.datasets.CIFAR10` | `mx.gluon.data.vision.CIFAR10`|
| CIFAR100 Dataset. | `torchvision.datasets.CIFAR100` | `mx.gluon.data.vision.CIFAR100` |
| A generic data loader where the images are arranged in folders. | `torchvision.datasets.ImageFolder(root, transform=None, target_transform=None, loader=<function default_loader>)` | `mx.gluon.data.vision.ImageFolderDataset(root, flag, transform=None)`|

### Serialization
Serialization and De-Serialization are achieved by calling `save_parameters` and `load_parameters`.

| Class               | Pytorch                           | MXNet Gluon                              |
|------------------------|-----------------------------------|------------------------------------------|
| Save model parameters | `torch.save(the_model.state_dict(), filename)`| `model.save_parameters(filename)`|
| Load parameters | `the_model.load_state_dict(torch.load(PATH))` | `model.load_parameters(filename, ctx, allow_missing=False, ignore_extra=False)` |
