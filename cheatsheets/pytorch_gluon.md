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
| Context scope          | Not available                     | `with mx.gpu(1):`<br/>&nbsp;&nbsp;&nbsp;&nbsp;`y = mx.nd.ones((3,5))`      |
###  Cross-device

## Autograd
### variable wrapper vs autograd scope
### gluon can supply head grad
### scope override (pause, train_mode, predict_mode)
### batch-end synchronization is needed

## Pytorch module and Gluon blocks
### usage of existing blocks look alike
### for new block definition, gluon needs name_scope
### HybridBlock can be hybridized, and allows partial-shape info
### SymbolBlock

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
