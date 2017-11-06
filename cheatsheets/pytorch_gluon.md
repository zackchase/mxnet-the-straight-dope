# PyTorch to MXNet
This cheatsheet serves as a quick reference for PyTorch users.

## Pytorch Tensor and MXNet NDArray
###  Tensor operation
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

### usage of existing blocks look alike

| Function               | PyTorch                           | MXNet Gluon                                                                |
|------------------------|-----------------------------------|----------------------------------------------------------------------------|
| Usage of existing blocks    |  `y=net(x)`  |  `y=net(x)`   |

### HybridBlock can be hybridized, and allows partial-shape info

HybridBlock supports forwarding with both Symbol and NDArray. After hybridized, HybridBlock will create a symbolic graph representing the forward computation and cache it.

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
### for gluon zero_grad only needed when `grad_req` is `'add'`
### Multi-GPU training
### Distributed training

## Monitoring
### MXNet has pre-defined metrics
### Data visualization

## I/O and deploy
### Data loading
### Serialization
