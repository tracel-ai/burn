# Tensor

As previously explained in the [model section](../basic-workflow/model.md), the Tensor struct has 3
generic arguments: the backend B, the dimensionality D, and the data type.

```rust , ignore
Tensor<B, D>           // Float tensor (default)
Tensor<B, D, Float>    // Explicit float tensor
Tensor<B, D, Int>      // Int tensor
Tensor<B, D, Bool>     // Bool tensor
```

Note that the specific element types used for `Float`, `Int`, and `Bool` tensors are defined by
backend implementations.

Burn Tensors are defined by the number of dimensions D in its declaration as opposed to its shape.
The actual shape of the tensor is inferred from its initialization. For example, a Tensor of size
(5,) is initialized as below:

```rust, ignore
let floats = [1.0, 2.0, 3.0, 4.0, 5.0];

// correct: Tensor is 1-Dimensional with 5 elements
let tensor_1 = Tensor::<Backend, 1>::from_floats(floats);

// incorrect: let tensor_1 = Tensor::<Backend, 5>::from_floats(floats);
// this will lead to an error and is for creating a 5-D tensor
```

### Initialization

Burn Tensors are primarily initialized using the `from_data()` method which takes the `Data` struct
as input. The `Data` struct has two fields: value & shape. To retrieve the data from a tensor, the
method `.to_data()` should be employed when intending to reuse the tensor afterward. Alternatively,
`.into_data()` is recommended for one-time use. Let's look at a couple of examples for initializing
a tensor from different inputs.

```rust, ignore

// Initialization from a given Backend (Wgpu)
let tensor_1 = Tensor::<Wgpu, 1>::from_data([1.0, 2.0, 3.0]);

// Initialization from a generic Backend
let tensor_2 = Tensor::<Backend, 1>::from_data(Data::from([1.0, 2.0, 3.0]).convert());

// Initialization using from_floats (Recommended for f32 ElementType)
// Will be converted to Data internally.
// `.convert()` not needed as from_floats() defined for fixed ElementType
let tensor_3 = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0]);

// Initialization of Int Tensor from array slices
let arr: [i32; 6] = [1, 2, 3, 4, 5, 6];
let tensor_4 = Tensor::<Backend, 1, Int>::from_data(Data::from(&arr[0..3]).convert());

// Initialization from a custom type

struct BodyMetrics {
    age: i8,
    height: i16,
    weight: f32
}

let bmi = BodyMetrics{
        age: 25,
        height: 180,
        weight: 80.0
    };
let data  = Data::from([bmi.age as f32, bmi.height as f32, bmi.weight]).convert();
let tensor_5 = Tensor::<Backend, 1>::from_data(data);

```

The `.convert()` method for Data struct is called to ensure that the data's primitive type is
consistent across all backends. With `.from_floats()` method the ElementType is fixed as f32 and
therefore no convert operation is required across backends. This operation can also be done at
element wise level as:
`let tensor_6 = Tensor::<B, 1, Int>::from_data(Data::from([(item.age as i64).elem()])`. The
`ElementConversion` trait however needs to be imported for the element wise operation.

## Ownership and Cloning

Almost all Burn operations take ownership of the input tensors. Therefore, reusing a tensor multiple
times will necessitate cloning it. Let's look at an example to understand the ownership rules and
cloning better. Suppose we want to do a simple min-max normalization of an input tensor.

```rust, ignore
let input = Tensor::<Wgpu, 1>::from_floats([1.0, 2.0, 3.0, 4.0]);
let min = input.min();
let max = input.max();
let input = (input - min).div(max - min);
```

With PyTorch tensors, the above code would work as expected. However, Rust's strict ownership rules
will give an error and prevent using the input tensor after the first `.min()` operation. The
ownership of the input tensor is transferred to the variable `min` and the input tensor is no longer
available for further operations. Burn Tensors like most complex primitives do not implement the
`Copy` trait and therefore have to be cloned explicitly. Now let's rewrite a working example of
doing min-max normalization with cloning.

```rust, ignore
let input = Tensor::<Wgpu, 1>::from_floats([1.0, 2.0, 3.0, 4.0]);
let min = input.clone().min();
let max = input.clone().max();
let input = (input.clone() - min.clone()).div(max - min);
println!("{:?}", input.to_data());// Success: [0.0, 0.33333334, 0.6666667, 1.0]

// Notice that max, min have been moved in last operation so
// the below print will give an error.
// If we want to use them for further operations,
// they will need to be cloned in similar fashion.
// println!("{:?}", min.to_data());
```

We don't need to be worried about memory overhead because with cloning, the tensor's buffer isn't
copied, and only a reference to it is increased. This makes it possible to determine exactly how
many times a tensor is used, which is very convenient for reusing tensor buffers or even fusing
operations into a single kernel ([burn-fusion](https://burn.dev/docs/burn_fusion/index.htmls)). For
that reason, we don't provide explicit inplace operations. If a tensor is used only one time,
inplace operations will always be used when available.

## Tensor Operations

Normally with PyTorch, explicit inplace operations aren't supported during the backward pass, making
them useful only for data preprocessing or inference-only model implementations. With Burn, you can
focus more on _what_ the model should do, rather than on _how_ to do it. We take the responsibility
of making your code run as fast as possible during training as well as inference. The same
principles apply to broadcasting; all operations support broadcasting unless specified otherwise.

Here, we provide a list of all supported operations along with their PyTorch equivalents. Note that
for the sake of simplicity, we ignore type signatures. For more details, refer to the
[full documentation](https://docs.rs/burn/latest/burn/tensor/struct.Tensor.html).

### Basic Operations

Those operations are available for all tensor kinds: `Int`, `Float`, and `Bool`.

| Burn                                  | PyTorch Equivalent                   |
| ------------------------------------- | ------------------------------------ |
| `Tensor::cat(tensors, dim)`           | `torch.cat(tensors, dim)`            |
| `Tensor::empty(shape, device)`        | `torch.empty(shape, device=device)`  |
| `Tensor::from_primitive(primitive)`   | N/A                                  |
| `Tensor::stack(tensors, dim)`         | `torch.stack(tensors, dim)`          |
| `tensor.all()`                        | `tensor.all()`                       |
| `tensor.all_dim(dim)`                 | `tensor.all(dim)`                    |
| `tensor.any()`                        | `tensor.any()`                       |
| `tensor.any_dim(dim)`                 | `tensor.any(dim)`                    |
| `tensor.expand(shape)`                | `tensor.expand(shape)`               |
| `tensor.chunk(num_chunks, dim)`       | `tensor.chunk(num_chunks, dim)`      |
| `tensor.device()`                     | `tensor.device`                      |
| `tensor.dims()`                       | `tensor.size()`                      |
| `tensor.equal(other)`                 | `x == y`                             |
| `tensor.flatten(start_dim, end_dim)`  | `tensor.flatten(start_dim, end_dim)` |
| `tensor.flip(axes)`                   | `tensor.flip(axes)`                  |
| `tensor.into_data()`                  | N/A                                  |
| `tensor.into_primitive()`             | N/A                                  |
| `tensor.into_scalar()`                | `tensor.item()`                      |
| `tensor.narrow(dim, start, length)`   | `tensor.narrow(dim, start, length)`  |
| `tensor.not_equal(other)`             | `x != y`                             |
| `tensor.permute(axes)`                | `tensor.permute(axes)`               |
| `tensor.repeat(2, 4)`                 | `tensor.repeat([1, 1, 4])`           |
| `tensor.reshape(shape)`               | `tensor.view(shape)`                 |
| `tensor.shape()`                      | `tensor.shape`                       |
| `tensor.slice(ranges)`                | `tensor[(*ranges,)]`                 |
| `tensor.slice_assign(ranges, values)` | `tensor[(*ranges,)] = values`        |
| `tensor.squeeze(dim)`                 | `tensor.squeeze(dim)`                |
| `tensor.to_data()`                    | N/A                                  |
| `tensor.to_device(device)`            | `tensor.to(device)`                  |
| `tensor.unsqueeze()`                  | `tensor.unsqueeze(0)`                |
| `tensor.unsqueeze_dim(dim)`           | `tensor.unsqueeze(dim)`              |

### Numeric Operations

Those operations are available for numeric tensor kinds: `Float` and `Int`.

| Burn                                                            | PyTorch Equivalent                             |
| --------------------------------------------------------------- | ---------------------------------------------- |
| `Tensor::eye(size, device)`                                     | `torch.eye(size, device=device)`               |
| `Tensor::full(shape, fill_value, device)`                       | `torch.full(shape, fill_value, device=device)` |
| `Tensor::ones(shape, device)`                                   | `torch.ones(shape, device=device)`             |
| `Tensor::zeros(shape)`                                          | `torch.zeros(shape)`                           |
| `Tensor::zeros(shape, device)`                                  | `torch.zeros(shape, device=device)`            |
| `tensor.abs()`                                                  | `torch.abs(tensor)`                            |
| `tensor.add(other)` or `tensor + other`                         | `tensor + other`                               |
| `tensor.add_scalar(scalar)` or `tensor + scalar`                | `tensor + scalar`                              |
| `tensor.all_close(other, atol, rtol)`                           | `torch.allclose(tensor, other, atol, rtol)`    |
| `tensor.argmax(dim)`                                            | `tensor.argmax(dim)`                           |
| `tensor.argmin(dim)`                                            | `tensor.argmin(dim)`                           |
| `tensor.bool()`                                                 | `tensor.bool()`                                |
| `tensor.clamp(min, max)`                                        | `torch.clamp(tensor, min=min, max=max)`        |
| `tensor.clamp_max(max)`                                         | `torch.clamp(tensor, max=max)`                 |
| `tensor.clamp_min(min)`                                         | `torch.clamp(tensor, min=min)`                 |
| `tensor.div(other)` or `tensor / other`                         | `tensor / other`                               |
| `tensor.div_scalar(scalar)` or `tensor / scalar`                | `tensor / scalar`                              |
| `tensor.equal_elem(other)`                                      | `tensor.eq(other)`                             |
| `tensor.gather(dim, indices)`                                   | `torch.gather(tensor, dim, indices)`           |
| `tensor.greater(other)`                                         | `tensor.gt(other)`                             |
| `tensor.greater_elem(scalar)`                                   | `tensor.gt(scalar)`                            |
| `tensor.greater_equal(other)`                                   | `tensor.ge(other)`                             |
| `tensor.greater_equal_elem(scalar)`                             | `tensor.ge(scalar)`                            |
| `tensor.is_close(other, atol, rtol)`                            | `torch.isclose(tensor, other, atol, rtol)`     |
| `tensor.lower(other)`                                           | `tensor.lt(other)`                             |
| `tensor.lower_elem(scalar)`                                     | `tensor.lt(scalar)`                            |
| `tensor.lower_equal(other)`                                     | `tensor.le(other)`                             |
| `tensor.lower_equal_elem(scalar)`                               | `tensor.le(scalar)`                            |
| `tensor.mask_fill(mask, value)`                                 | `tensor.masked_fill(mask, value)`              |
| `tensor.mask_where(mask, value_tensor)`                         | `torch.where(mask, value_tensor, tensor)`      |
| `tensor.max()`                                                  | `tensor.max()`                                 |
| `tensor.max_dim(dim)`                                           | `tensor.max(dim, keepdim=True)`                |
| `tensor.max_dim_with_indices(dim)`                              | N/A                                            |
| `tensor.max_pair(other)`                                        | `torch.Tensor.max(a,b)`                        |
| `tensor.mean()`                                                 | `tensor.mean()`                                |
| `tensor.mean_dim(dim)`                                          | `tensor.mean(dim, keepdim=True)`               |
| `tensor.min()`                                                  | `tensor.min()`                                 |
| `tensor.min_dim(dim)`                                           | `tensor.min(dim, keepdim=True)`                |
| `tensor.min_dim_with_indices(dim)`                              | N/A                                            |
| `tensor.min_pair(other)`                                        | `torch.Tensor.min(a,b)`                        |
| `tensor.mul(other)` or `tensor * other`                         | `tensor * other`                               |
| `tensor.mul_scalar(scalar)` or `tensor * scalar`                | `tensor * scalar`                              |
| `tensor.neg()` or `-tensor`                                     | `-tensor`                                      |
| `tensor.not_equal_elem(scalar)`                                 | `tensor.ne(scalar)`                            |
| `tensor.powf(other)` or `tensor.powi(intother)`                 | `tensor.pow(other)`                            |
| `tensor.powf_scalar(scalar)` or `tensor.powi_scalar(intscalar)` | `tensor.pow(scalar)`                           |
| `tensor.prod()`                                                 | `tensor.prod()`                                |
| `tensor.prod_dim(dim)`                                          | `tensor.prod(dim, keepdim=True)`               |
| `tensor.scatter(dim, indices, values)`                          | `tensor.scatter_add(dim, indices, values)`     |
| `tensor.select(dim, indices)`                                   | `tensor.index_select(dim, indices)`            |
| `tensor.select_assign(dim, indices, values)`                    | N/A                                            |
| `tensor.sign()`                                                 | `tensor.sign()`                                |
| `tensor.sub(other)` or `tensor - other`                         | `tensor - other`                               |
| `tensor.sub_scalar(scalar)` or `tensor - scalar`                | `tensor - scalar`                              |
| `tensor.sum()`                                                  | `tensor.sum()`                                 |
| `tensor.sum_dim(dim)`                                           | `tensor.sum(dim, keepdim=True)`                |
| `tensor.tril(diagonal)`                                         | `torch.tril(tensor, diagonal)`                 |
| `tensor.triu(diagonal)`                                         | `torch.triu(tensor, diagonal)`                 |
| `tensor.sort(dim)`                                              | `tensor.sort(dim).values`                      |
| `tensor.sort_descending(dim)`                                   | `tensor.sort(dim, descending=True).values`     |
| `tensor.sort_with_indices(dim)`                                 | `tensor.sort(dim)`                             |
| `tensor.sort_descending_with_indices(dim)`                      | `tensor.sort(dim, descending=True)`            |
| `tensor.argsort(dim)`                                           | `tensor.argsort(dim)`                          |
| `tensor.argsort_descending(dim)`                                | `tensor.argsort(dim, descending=True)`         |
| `tensor.topk(k, dim)`                                           | `tensor.topk(k, dim).values`                   |
| `tensor.topk_with_indices(k, dim)`                              | `tensor.topk(k, dim)`                          |

### Float Operations

Those operations are only available for `Float` tensors.

| Burn API                                     | PyTorch Equivalent                 |
| -------------------------------------------- | ---------------------------------- |
| `tensor.cos()`                               | `tensor.cos()`                     |
| `tensor.erf()`                               | `tensor.erf()`                     |
| `tensor.exp()`                               | `tensor.exp()`                     |
| `tensor.from_floats(floats, device)`         | N/A                                |
| `tensor.from_full_precision(tensor)`         | N/A                                |
| `tensor.int()`                               | Similar to `tensor.to(torch.long)` |
| `tensor.log()`                               | `tensor.log()`                     |
| `tensor.log1p()`                             | `tensor.log1p()`                   |
| `tensor.matmul(other)`                       | `tensor.matmul(other)`             |
| `tensor.one_hot(index, num_classes, device)` | N/A                                |
| `tensor.ones_like()`                         | `torch.ones_like(tensor)`          |
| `tensor.random(shape, distribution, device)` | N/A                                |
| `tensor.random_like(distribution)`           | `torch.rand_like()` only uniform   |
| `tensor.recip()`                             | `tensor.reciprocal()`              |
| `tensor.sin()`                               | `tensor.sin()`                     |
| `tensor.sqrt()`                              | `tensor.sqrt()`                    |
| `tensor.swap_dims(dim1, dim2)`               | `tensor.transpose(dim1, dim2)`     |
| `tensor.tanh()`                              | `tensor.tanh()`                    |
| `tensor.to_full_precision()`                 | `tensor.to(torch.float)`           |
| `tensor.transpose()`                         | `tensor.T`                         |
| `tensor.var(dim)`                            | `tensor.var(dim)`                  |
| `tensor.var_bias(dim)`                       | N/A                                |
| `tensor.var_mean(dim)`                       | N/A                                |
| `tensor.var_mean_bias(dim)`                  | N/A                                |
| `tensor.zeros_like()`                        | `torch.zeros_like(tensor)`         |

# Int Operations

Those operations are only available for `Int` tensors.

| Burn API                                         | PyTorch Equivalent                                      |
| ------------------------------------------------ | ------------------------------------------------------- |
| `tensor.arange(5..10, device)       `            | `tensor.arange(start=5, end=10, device=device)`         |
| `tensor.arange_step(5..10, 2, device)`           | `tensor.arange(start=5, end=10, step=2, device=device)` |
| `tensor.float()`                                 | `tensor.to(torch.float)`                                |
| `tensor.from_ints(ints)`                         | N/A                                                     |
| `tensor.int_random(shape, distribution, device)` | N/A                                                     |

# Bool Operations

Those operations are only available for `Bool` tensors.

| Burn API                            | PyTorch Equivalent              |
| ----------------------------------- | ------------------------------- |
| `Tensor.diag_mask(shape, diagonal)` | N/A                             |
| `Tensor.tril_mask(shape, diagonal)` | N/A                             |
| `Tensor.triu_mask(shape, diagonal)` | N/A                             |
| `tensor.argwhere()`                 | `tensor.argwhere()`             |
| `tensor.float()`                    | `tensor.to(torch.float)`        |
| `tensor.int()`                      | `tensor.to(torch.long)`         |
| `tensor.nonzero()`                  | `tensor.nonzero(as_tuple=True)` |
| `tensor.not()`                      | `tensor.logical_not()`          |

## Activation Functions

| Burn API                                 | PyTorch Equivalent                         |
| ---------------------------------------- | ------------------------------------------ |
| `activation::gelu(tensor)`               | `nn.functional.gelu(tensor)`               |
| `activation::log_sigmoid(tensor)`        | `nn.functional.log_sigmoid(tensor)`        |
| `activation::log_softmax(tensor, dim)`   | `nn.functional.log_softmax(tensor, dim)`   |
| `activation::mish(tensor)`               | `nn.functional.mish(tensor)`               |
| `activation::prelu(tensor,alpha)`        | `nn.functional.prelu(tensor,weight)`       |
| `activation::quiet_softmax(tensor, dim)` | `nn.functional.quiet_softmax(tensor, dim)` |
| `activation::relu(tensor)`               | `nn.functional.relu(tensor)`               |
| `activation::sigmoid(tensor)`            | `nn.functional.sigmoid(tensor)`            |
| `activation::silu(tensor)`               | `nn.functional.silu(tensor)`               |
| `activation::softmax(tensor, dim)`       | `nn.functional.softmax(tensor, dim)`       |
| `activation::softplus(tensor, beta)`     | `nn.functional.softplus(tensor, beta)`     |
| `activation::tanh(tensor)`               | `nn.functional.tanh(tensor)`               |
