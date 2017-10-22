# PyTorch to MXNet
This cheatsheet serves as a quick reference for PyTorch users.

## Pytorch Tensor and MXNet NDArray
###  Tensor operation
We document PyTorch function namesthat are different than MXNet NDArray

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
| Gathers values along an axis specified by dim | `x.gather(dim,  torch.LongTensor([[0,0],[1,0]]))` | Not available                     |
| Limits the values of a tensor to between min and max | `x.clamp(min, max)`| `nd.clip(x, min, max)`                                    |
| Returns a copy of the tensor  | `x.clone()`                               | `x.copy()`                                                |
| Cross product                 | `x.cross(y)`                              | Not available                                             |
| Cumulative product along an axis| `x.cumprod(1)`                          | Not available                                             |
| Cumulative sum along an axis  | `x.cumsum(1)`                             | Not available                                             |
| Address of the first element  | `x.data_ptr()`                            | Not available                                             |
| Creates a diagonal tensor     | `x.diag()`                                | Not available                                             |
| Device context of a tensor    | `print(x)` will print which device x is on| `x.context`                                               |
| Data type of a tensor         | `x.type()`                                | `x.dtype`                                                 |
| Returns the shape of a tensor | `x.size()`                                | `x.shape`                                                 |
| Number of elements in a tensor| `x.numel()`                               | `x.size`                                                  |
| Returns this tensor as a NumPy ndarray | `x.numpy()`                      | `x.asnumpy()`                                             |
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
