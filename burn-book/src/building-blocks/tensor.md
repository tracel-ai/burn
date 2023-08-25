# Tensor

As previously explained in the [model section](../basic-workflow/model.md), the Tensor struct has 3 generic arguments: the backend, the dimension number (rank), and the kind.

```rust , ignore
Tensor<B, D>           // Float tensor (default)
Tensor<B, D, Float>    // Explicit float tensor
Tensor<B, D, Int>      // Int tensor
Tensor<B, D, Bool>     // Bool tensor
```

Note that the specific element types used for `Float`, `Int`, and `Bool` tensors are defined by backend implementations.

## Operations

Almost all Burn operations take ownership of the input tensors.
Therefore, reusing a tensor multiple times will necessitate cloning it.
Don't worry, the tensor's buffer isn't copied, but a reference to it is increased.
This makes it possible to determine exactly how many times a tensor is used, which is very convenient for reusing tensor buffers and improving performance.
For that reason, we don't provide explicit inplace operations.
If a tensor is used only one time, inplace operations will always be used when available.

Normally with PyTorch, explicit inplace operations aren't supported during the backward pass, making them useful only for data preprocessing or inference-only model implementations.
With Burn, you can focus more on _what_ the model should do, rather than on _how_ to do it.
We take the responsibility of making your code run as fast as possible during training as well as inference.
The same principles apply to broadcasting; all operations support broadcasting unless specified otherwise.

Here, we provide a list of all supported operations along with their PyTorch equivalents.
Note that for the sake of simplicity, we ignore type signatures.
For more details, refer to the [full documentation](https://docs.rs/burn/latest/burn/tensor/struct.Tensor.html).

### Basic Operations

Those operations are available for all tensor kinds: `Int`, `Float`, and `Bool`.

| Burn                                                  | PyTorch Equivalent                                |
|-------------------------------------------------------|---------------------------------------------------|
| `Tensor::empty(shape)`                                | `torch.empty(shape)`                              |
| `Tensor::empty_device(shape, device)`                 | `torch.empty(shape, device=device)`               |
| `tensor.dims()`                                       | `tensor.size()`                                   |
| `tensor.shape()`                                      | `tensor.shape`                                    |
| `tensor.reshape(shape)`                               | `tensor.view(shape)`                              |
| `tensor.flatten(start_dim, end_dim)`                  | `tensor.flatten(start_dim, end_dim)`              |
| `tensor.squeeze(dim)`                                 | `tensor.squeeze(dim)`                             |
| `tensor.unsqueeze()`                                  | `tensor.unsqueeze(0)`                             |
| `tensor.slice(ranges)`                                | `tensor[(*ranges,)]`                              |
| `tensor.slice_assign(ranges, values)`                 | `tensor[(*ranges,)] = values`                     |
| `tensor.device()`                                     | `tensor.device`                                   |
| `tensor.to_device(device)`                            | `tensor.to(device)`                               |
| `tensor.repeat(2, 4)`                                 | `tensor.repeat([1, 1, 4])`                        |
| `tensor.equal(other)`                                 | `x == y`                                          |
| `Tensor::cat(tensors, dim)`                           | `torch.cat(tensors, dim)`                         |
| `tensor.into_data()`                                  | N/A                                               |
| `tensor.to_data()`                                    | N/A                                               |
| `Tensor::from_data(data)`                             | N/A                                               |
| `Tensor::from_data_device(data, device)`              | N/A                                               |
| `tensor.into_primitive()`                             | N/A                                               |
| `Tensor::from_primitive(primitive)`                   | N/A                                               |

### Numeric Operations

Those operations are available for numeric tensor kinds: `Float` and `Int`.

| Burn                                               | PyTorch Equivalent                                      |
|----------------------------------------------------|---------------------------------------------------------|
| `tensor.into_scalar()`                             | `tensor.item()` (for single-element tensors)            |
| `tensor + other` or `tensor.add(other)`            | `tensor + other`                                        |
| `tensor + scalar` or `tensor.add_scalar(scalar)`   | `tensor + scalar`                                       |
| `tensor - other` or `tensor.sub(other)`            | `tensor - other`                                        |
| `tensor - scalar` or `tensor.sub_scalar(scalar)`   | `tensor - scalar`                                       |
| `tensor / other` or `tensor.div(other)`            | `tensor / other`                                        |
| `tensor / scalar` or `tensor.div_scalar(scalar)`   | `tensor / scalar`                                       |
| `tensor * other` or `tensor.mul(other)`            | `tensor * other`                                        |
| `tensor * scalar` or `tensor.mul_scalar(scalar)`   | `tensor * scalar`                                       |
| `-tensor` or `tensor.neg()`                        | `-tensor`                                               |
| `Tensor::zeros(shape)`                             | `torch.zeros(shape)`                                    |
| `Tensor::zeros_device(shape, device)`              | `torch.zeros(shape, device=device)`                     |
| `Tensor::ones(shape)`                              | `torch.ones(shape)`                                     |
| `Tensor::ones_device(shape, device)`               | `torch.ones(shape, device=device)`                      |
| `Tensor::full(shape, fill_value)`                  | `torch.full(shape, fill_value)`                         |
| `Tensor::full_device(shape, fill_value, device)`   | `torch.full(shape, fill_value, device=device)`          |
| `tensor.mean()`                                    | `tensor.mean()`                                         |
| `tensor.sum()`                                     | `tensor.sum()`                                          |
| `tensor.mean_dim(dim)`                             | `tensor.mean(dim)`                                      |
| `tensor.sum_dim(dim)`                              | `tensor.sum(dim)`                                       |
| `tensor.equal_elem(other)`                         | `tensor.eq(other)`                                      |
| `tensor.greater(other)`                            | `tensor.gt(other)`                                      |
| `tensor.greater_elem(scalar)`                      | `tensor.gt(scalar)`                                     |
| `tensor.greater_equal(other)`                      | `tensor.ge(other)`                                      |
| `tensor.greater_equal_elem(scalar)`                | `tensor.ge(scalar)`                                     |
| `tensor.lower(other)`                              | `tensor.lt(other)`                                      |
| `tensor.lower_elem(scalar)`                        | `tensor.lt(scalar)`                                     |
| `tensor.lower_equal(other)`                        | `tensor.le(other)`                                      |
| `tensor.lower_equal_elem(scalar)`                  | `tensor.le(scalar)`                                     |
| `tensor.mask_where(mask, value_tensor)`            | `torch.where(mask, value_tensor, tensor)`               |
| `tensor.mask_fill(mask, value)`                    | `tensor.masked_fill(mask, value)`                       |
| `tensor.gather(dim, indices)`                      | `torch.gather(tensor, dim, indices)`                    |
| `tensor.scatter(dim, indices, values)`             | `tensor.scatter_add(dim, indices, values)`              |
| `tensor.select(dim, indices)`                      | `tensor.index_select(dim, indices)`                     |
| `tensor.select_assign(dim, indices, values)`       | N/A                                                     |
| `tensor.argmax(dim)`                               | `tensor.argmax(dim)`                                    |
| `tensor.max()`                                     | `tensor.max()`                                          |
| `tensor.max_dim(dim)`                              | `tensor.max(dim)`                                       |
| `tensor.max_dim_with_indices(dim)`                 | N/A                                                     |
| `tensor.argmin(dim)`                               | `tensor.argmin(dim)`                                    |
| `tensor.min()`                                     | `tensor.min()`                                          |
| `tensor.min_dim(dim)`                              | `tensor.min(dim)`                                       |
| `tensor.min_dim_with_indices(dim)`                 | N/A                                                     |
| `tensor.clamp(min, max)`                           | `torch.clamp(tensor, min=min, max=max)`                 |
| `tensor.clamp_min(min)`                            | `torch.clamp(tensor, min=min)`                          |
| `tensor.clamp_max(max)`                            | `torch.clamp(tensor, max=max)`                          |
| `tensor.abs()`                                     | `torch.abs(tensor)`                                     |

### Float Operations

Those operations are only available for `Float` tensors.

| Burn API                                               | PyTorch Equivalent                                |
|--------------------------------------------------------|---------------------------------------------------|
| `tensor.exp()`                                         | `tensor.exp()`                                    |
| `tensor.log()`                                         | `tensor.log()`                                    |
| `tensor.log1p()`                                       | `tensor.log1p()`                                  |
| `tensor.erf()`                                         | `tensor.erf()`                                    |
| `tensor.powf(value)`                                   | `tensor.pow(value)`                               |
| `tensor.sqrt()`                                        | `tensor.sqrt()`                                   |
| `tensor.cos()`                                         | `tensor.cos()`                                    |
| `tensor.sin()`                                         | `tensor.sin()`                                    |
| `tensor.tanh()`                                        | `tensor.tanh()`                                   |
| `tensor.from_floats(floats)`                           | N/A                                               |
| `tensor.int()`                                         | Similar to `tensor.to(torch.long)`                |
| `tensor.zeros_like()`                                  | `torch.zeros_like(tensor)`                        |
| `tensor.ones_like()`                                   | `torch.ones_like(tensor)`                         |
| `tensor.random_like(distribution)`                     | `torch.rand_like()` only uniform                  |
| `tensor.one_hot(index, num_classes)`                   | N/A                                               |
| `tensor.transpose()`                                   | `tensor.T`                                        |
| `tensor.swap_dims(dim1, dim2)`                         | `tensor.transpose(dim1, dim2)`                    |
| `tensor.matmul(other)`                                 | `tensor.matmul(other)`                            |
| `tensor.var(dim)`                                      | `tensor.var(dim)`                                 |
| `tensor.var_bias(dim)`                                 | N/A                                               |
| `tensor.var_mean(dim)`                                 | N/A                                               |
| `tensor.var_mean_bias(dim)`                            | N/A                                               |
| `tensor.random(shape, distribution)`                   | N/A                                               |
| `tensor.random_device(shape, distribution, device)`    | N/A                                               |
| `tensor.to_full_precision()`                           | `tensor.to(torch.float)`                          |
| `tensor.from_full_precision(tensor)`                   | N/A                                               |


# Int Operations

Those operations are only available for `Int` tensors.

| Burn API                                               | PyTorch Equivalent                                      |
|--------------------------------------------------------|---------------------------------------------------------|
| `tensor.from_ints(ints)`                               | N/A                                                     |
| `tensor.float()`                                       | Similar to `tensor.to(torch.float)`                     |
| `tensor.arange(5..10)`                                 | `tensor.arange(start=5, end=10)`                        |
| `tensor.arange_device(5..10, device)`                  | `tensor.arange(start=5, end=10, device=device)`         |
| `tensor.arange_step(5..10, 2)`                         | `tensor.arange(start=5, end=10, step=2)`                |
| `tensor.arange_step_device(5..10, 2, device)`          | `tensor.arange(start=5, end=10, step=2, device=device)` |


# Bool Operations

Those operations are only available for `Bool` tensors.

| Burn API                                               | PyTorch Equivalent                                      |
|--------------------------------------------------------|---------------------------------------------------------|
| `tensor.float()`                                       | Similar to `tensor.to(torch.float)`                     |
| `tensor.int()`                                         | Similar to `tensor.to(torch.long)`                      |
