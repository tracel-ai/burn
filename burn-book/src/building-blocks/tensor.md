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

// Get the default device
let device = Default::default();

// correct: Tensor is 1-Dimensional with 5 elements
let tensor_1 = Tensor::<Backend, 1>::from_floats(floats, &device);

// incorrect: let tensor_1 = Tensor::<Backend, 5>::from_floats(floats, &device);
// this will lead to an error and is for creating a 5-D tensor
```

### Initialization

Burn Tensors are primarily initialized using the `from_data()` method which takes the `TensorData`
struct as input. The `TensorData` struct has two public fields: `shape` and `dtype`. The `value`,
now stored as bytes, is private but can be accessed via any of the following methods: `as_slice`,
`as_mut_slice`, `to_vec` and `iter`. To retrieve the data from a tensor, the method `.to_data()`
should be employed when intending to reuse the tensor afterward. Alternatively, `.into_data()` is
recommended for one-time use. Let's look at a couple of examples for initializing a tensor from
different inputs.

```rust, ignore

// Initialization from a given Backend (Wgpu)
let tensor_1 = Tensor::<Wgpu, 1>::from_data([1.0, 2.0, 3.0], &device);

// Initialization from a generic Backend
let tensor_2 = Tensor::<Backend, 1>::from_data(TensorData::from([1.0, 2.0, 3.0]), &device);

// Initialization using from_floats (Recommended for f32 ElementType)
// Will be converted to TensorData internally.
let tensor_3 = Tensor::<Backend, 1>::from_floats([1.0, 2.0, 3.0], &device);

// Initialization of Int Tensor from array slices
let arr: [i32; 6] = [1, 2, 3, 4, 5, 6];
let tensor_4 = Tensor::<Backend, 1, Int>::from_data(TensorData::from(&arr[0..3]), &device);

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
let data  = TensorData::from([bmi.age as f32, bmi.height as f32, bmi.weight]);
let tensor_5 = Tensor::<Backend, 1>::from_data(data, &device);

```

## Ownership and Cloning

Almost all Burn operations take ownership of the input tensors. Therefore, reusing a tensor multiple
times will necessitate cloning it. Let's look at an example to understand the ownership rules and
cloning better. Suppose we want to do a simple min-max normalization of an input tensor.

```rust, ignore
let input = Tensor::<Wgpu, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
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
let input = Tensor::<Wgpu, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
let min = input.clone().min();
let max = input.clone().max();
let input = (input.clone() - min.clone()).div(max - min);
println!("{}", input.to_data());// Success: [0.0, 0.33333334, 0.6666667, 1.0]

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

| Burn                                        | PyTorch Equivalent                                                        |
| ------------------------------------------- | ------------------------------------------------------------------------- |
| `Tensor::cat(tensors, dim)`                 | `torch.cat(tensors, dim)`                                                 |
| `Tensor::empty(shape, device)`              | `torch.empty(shape, device=device)`                                       |
| `Tensor::from_primitive(primitive)`         | N/A                                                                       |
| `Tensor::stack(tensors, dim)`               | `torch.stack(tensors, dim)`                                               |
| `tensor.all()`                              | `tensor.all()`                                                            |
| `tensor.all_dim(dim)`                       | `tensor.all(dim)`                                                         |
| `tensor.any()`                              | `tensor.any()`                                                            |
| `tensor.any_dim(dim)`                       | `tensor.any(dim)`                                                         |
| `tensor.chunk(num_chunks, dim)`             | `tensor.chunk(num_chunks, dim)`                                           |
| `tensor.split(split_size, dim)`             | `tensor.split(split_size, dim)`                                           |
| `tensor.split_with_sizes(split_sizes, dim)` | `tensor.split([split_sizes], dim)`                                        |
| `tensor.device()`                           | `tensor.device`                                                           |
| `tensor.dtype()`                            | `tensor.dtype`                                                            |
| `tensor.dims()`                             | `tensor.size()`                                                           |
| `tensor.equal(other)`                       | `x == y`                                                                  |
| `tensor.expand(shape)`                      | `tensor.expand(shape)`                                                    |
| `tensor.flatten(start_dim, end_dim)`        | `tensor.flatten(start_dim, end_dim)`                                      |
| `tensor.flip(axes)`                         | `tensor.flip(axes)`                                                       |
| `tensor.into_data()`                        | N/A                                                                       |
| `tensor.into_primitive()`                   | N/A                                                                       |
| `tensor.into_scalar()`                      | `tensor.item()`                                                           |
| `tensor.narrow(dim, start, length)`         | `tensor.narrow(dim, start, length)`                                       |
| `tensor.not_equal(other)`                   | `x != y`                                                                  |
| `tensor.permute(axes)`                      | `tensor.permute(axes)`                                                    |
| `tensor.movedim(src, dst)`                  | `tensor.movedim(src, dst)`                                                |
| `tensor.repeat_dim(dim, times)`             | `tensor.repeat(*[times if i == dim else 1 for i in range(tensor.dim())])` |
| `tensor.repeat(sizes)`                      | `tensor.repeat(sizes)`                                                    |
| `tensor.reshape(shape)`                     | `tensor.view(shape)`                                                      |
| `tensor.shape()`                            | `tensor.shape`                                                            |
| `tensor.slice(ranges)`                      | `tensor[(*ranges,)]`                                                      |
| `tensor.slice_assign(ranges, values)`       | `tensor[(*ranges,)] = values`                                             |
| `tensor.squeeze(dim)`                       | `tensor.squeeze(dim)`                                                     |
| `tensor.swap_dims(dim1, dim2)`              | `tensor.transpose(dim1, dim2)`                                            |
| `tensor.to_data()`                          | N/A                                                                       |
| `tensor.to_device(device)`                  | `tensor.to(device)`                                                       |
| `tensor.transpose()`                        | `tensor.T`                                                                |
| `tensor.unsqueeze()`                        | `tensor.unsqueeze(0)`                                                     |
| `tensor.unsqueeze_dim(dim)`                 | `tensor.unsqueeze(dim)`                                                   |
| `tensor.unsqueeze_dims(dims)`               | N/A                                                                       |

### Numeric Operations

Those operations are available for numeric tensor kinds: `Float` and `Int`.

| Burn                                                            | PyTorch Equivalent                             |
| --------------------------------------------------------------- | ---------------------------------------------- |
| `Tensor::eye(size, device)`                                     | `torch.eye(size, device=device)`               |
| `Tensor::full(shape, fill_value, device)`                       | `torch.full(shape, fill_value, device=device)` |
| `Tensor::ones(shape, device)`                                   | `torch.ones(shape, device=device)`             |
| `Tensor::zeros(shape, device)`                                  | `torch.zeros(shape, device=device)`            |
| `tensor.abs()`                                                  | `torch.abs(tensor)`                            |
| `tensor.add(other)` or `tensor + other`                         | `tensor + other`                               |
| `tensor.add_scalar(scalar)` or `tensor + scalar`                | `tensor + scalar`                              |
| `tensor.all_close(other, atol, rtol)`                           | `torch.allclose(tensor, other, atol, rtol)`    |
| `tensor.argmax(dim)`                                            | `tensor.argmax(dim)`                           |
| `tensor.argmin(dim)`                                            | `tensor.argmin(dim)`                           |
| `tensor.argsort(dim)`                                           | `tensor.argsort(dim)`                          |
| `tensor.argsort_descending(dim)`                                | `tensor.argsort(dim, descending=True)`         |
| `tensor.bool()`                                                 | `tensor.bool()`                                |
| `tensor.clamp(min, max)`                                        | `torch.clamp(tensor, min=min, max=max)`        |
| `tensor.clamp_max(max)`                                         | `torch.clamp(tensor, max=max)`                 |
| `tensor.clamp_min(min)`                                         | `torch.clamp(tensor, min=min)`                 |
| `tensor.contains_nan()`                                         | N/A                                            |
| `tensor.div(other)` or `tensor / other`                         | `tensor / other`                               |
| `tensor.div_scalar(scalar)` or `tensor / scalar`                | `tensor / scalar`                              |
| `tensor.equal_elem(other)`                                      | `tensor.eq(other)`                             |
| `tensor.full_like(fill_value)`                                  | `torch.full_like(tensor, fill_value)           |
| `tensor.gather(dim, indices)`                                   | `torch.gather(tensor, dim, indices)`           |
| `tensor.greater(other)`                                         | `tensor.gt(other)`                             |
| `tensor.greater_elem(scalar)`                                   | `tensor.gt(scalar)`                            |
| `tensor.greater_equal(other)`                                   | `tensor.ge(other)`                             |
| `tensor.greater_equal_elem(scalar)`                             | `tensor.ge(scalar)`                            |
| `tensor.is_close(other, atol, rtol)`                            | `torch.isclose(tensor, other, atol, rtol)`     |
| `tensor.is_nan()`                                               | `torch.isnan(tensor)`                          |
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
| `tensor.ones_like()`                                            | `torch.ones_like(tensor)`                      |
| `tensor.one_hot(num_classes)`                                   | `torch.nn.functional.one_hot`                  |
| `tensor.one_hot_fill(num_classes, on_value, off_value, axis)`   | N/A                                            |
| `tensor.pad(pads, value)`                                       | `torch.nn.functional.pad(input, pad, value)`   |
| `tensor.powf(other)` or `tensor.powi(intother)`                 | `tensor.pow(other)`                            |
| `tensor.powf_scalar(scalar)` or `tensor.powi_scalar(intscalar)` | `tensor.pow(scalar)`                           |
| `tensor.prod()`                                                 | `tensor.prod()`                                |
| `tensor.prod_dim(dim)`                                          | `tensor.prod(dim, keepdim=True)`               |
| `tensor.rem(other)` or `tensor % other`                         | `tensor % other`                               |
| `tensor.scatter(dim, indices, values)`                          | `tensor.scatter_add(dim, indices, values)`     |
| `tensor.select(dim, indices)`                                   | `tensor.index_select(dim, indices)`            |
| `tensor.select_assign(dim, indices, values)`                    | N/A                                            |
| `tensor.sign()`                                                 | `tensor.sign()`                                |
| `tensor.sort(dim)`                                              | `tensor.sort(dim).values`                      |
| `tensor.sort_descending(dim)`                                   | `tensor.sort(dim, descending=True).values`     |
| `tensor.sort_descending_with_indices(dim)`                      | `tensor.sort(dim, descending=True)`            |
| `tensor.sort_with_indices(dim)`                                 | `tensor.sort(dim)`                             |
| `tensor.sub(other)` or `tensor - other`                         | `tensor - other`                               |
| `tensor.sub_scalar(scalar)` or `tensor - scalar`                | `tensor - scalar`                              |
| `tensor.sum()`                                                  | `tensor.sum()`                                 |
| `tensor.sum_dim(dim)`                                           | `tensor.sum(dim, keepdim=True)`                |
| `tensor.topk(k, dim)`                                           | `tensor.topk(k, dim).values`                   |
| `tensor.topk_with_indices(k, dim)`                              | `tensor.topk(k, dim)`                          |
| `tensor.tril(diagonal)`                                         | `torch.tril(tensor, diagonal)`                 |
| `tensor.triu(diagonal)`                                         | `torch.triu(tensor, diagonal)`                 |
| `tensor.zeros_like()`                                           | `torch.zeros_like(tensor)`                     |

### Float Operations

Those operations are only available for `Float` tensors.

| Burn API                                     | PyTorch Equivalent                 |
| -------------------------------------------- | ---------------------------------- |
| `tensor.cast(dtype)`                         | `tensor.to(dtype)`                 |
| `tensor.ceil()`                              | `tensor.ceil()`                    |
| `tensor.cos()`                               | `tensor.cos()`                     |
| `tensor.erf()`                               | `tensor.erf()`                     |
| `tensor.exp()`                               | `tensor.exp()`                     |
| `tensor.floor()`                             | `tensor.floor()`                   |
| `tensor.from_floats(floats, device)`         | N/A                                |
| `tensor.from_full_precision(tensor)`         | N/A                                |
| `tensor.int()`                               | Similar to `tensor.to(torch.long)` |
| `tensor.log()`                               | `tensor.log()`                     |
| `tensor.log1p()`                             | `tensor.log1p()`                   |
| `tensor.matmul(other)`                       | `tensor.matmul(other)`             |
| `tensor.random(shape, distribution, device)` | N/A                                |
| `tensor.random_like(distribution)`           | `torch.rand_like()` only uniform   |
| `tensor.recip()`                             | `tensor.reciprocal()`              |
| `tensor.round()`                             | `tensor.round()`                   |
| `tensor.sin()`                               | `tensor.sin()`                     |
| `tensor.sqrt()`                              | `tensor.sqrt()`                    |
| `tensor.tanh()`                              | `tensor.tanh()`                    |
| `tensor.to_full_precision()`                 | `tensor.to(torch.float)`           |
| `tensor.var(dim)`                            | `tensor.var(dim)`                  |
| `tensor.var_bias(dim)`                       | N/A                                |
| `tensor.var_mean(dim)`                       | N/A                                |
| `tensor.var_mean_bias(dim)`                  | N/A                                |

### Int Operations

Those operations are only available for `Int` tensors.

| Burn API                                         | PyTorch Equivalent                                      |
| ------------------------------------------------ | ------------------------------------------------------- |
| `Tensor::arange(5..10, device)`                  | `tensor.arange(start=5, end=10, device=device)`         |
| `Tensor::arange_step(5..10, 2, device)`          | `tensor.arange(start=5, end=10, step=2, device=device)` |
| `tensor.bitwise_and(other)`                      | `torch.bitwise_and(tensor, other)`                      |
| `tensor.bitwise_and_scalar(scalar)`              | `torch.bitwise_and(tensor, scalar)`                     |
| `tensor.bitwise_not()`                           | `torch.bitwise_not(tensor)`                             |
| `tensor.bitwise_left_shift(other)`               | `torch.bitwise_left_shift(tensor, other)`               |
| `tensor.bitwise_left_shift_scalar(scalar)`       | `torch.bitwise_left_shift(tensor, scalar)`              |
| `tensor.bitwise_right_shift(other)`              | `torch.bitwise_right_shift(tensor, other)`              |
| `tensor.bitwise_right_shift_scalar(scalar)`      | `torch.bitwise_right_shift(tensor, scalar)`             |
| `tensor.bitwise_or(other)`                       | `torch.bitwise_or(tensor, other)`                       |
| `tensor.bitwise_or_scalar(scalar)`               | `torch.bitwise_or(tensor, scalar)`                      |
| `tensor.bitwise_xor(other)`                      | `torch.bitwise_xor(tensor, other)`                      |
| `tensor.bitwise_xor_scalar(scalar)`              | `torch.bitwise_xor(tensor, scalar)`                     |
| `tensor.float()`                                 | `tensor.to(torch.float)`                                |
| `tensor.from_ints(ints)`                         | N/A                                                     |
| `tensor.int_random(shape, distribution, device)` | N/A                                                     |
| `tensor.cartesian_grid(shape, device)`           | N/A                                                     |

### Bool Operations

Those operations are only available for `Bool` tensors.

| Burn API                             | PyTorch Equivalent              |
| ------------------------------------ | ------------------------------- |
| `Tensor::diag_mask(shape, diagonal)` | N/A                             |
| `Tensor::tril_mask(shape, diagonal)` | N/A                             |
| `Tensor::triu_mask(shape, diagonal)` | N/A                             |
| `tensor.argwhere()`                  | `tensor.argwhere()`             |
| `tensor.float()`                     | `tensor.to(torch.float)`        |
| `tensor.int()`                       | `tensor.to(torch.long)`         |
| `tensor.nonzero()`                   | `tensor.nonzero(as_tuple=True)` |
| `tensor.not()`                       | `tensor.logical_not()`          |

### Quantization Operations

Those operations are only available for `Float` tensors on backends that implement quantization
strategies.

| Burn API                           | PyTorch Equivalent |
| ---------------------------------- | ------------------ |
| `tensor.quantize(scheme, qparams)` | N/A                |
| `tensor.dequantize()`              | N/A                |

## Activation Functions

| Burn API                                         | PyTorch Equivalent                                 |
| ------------------------------------------------ | -------------------------------------------------- |
| `activation::gelu(tensor)`                       | `nn.functional.gelu(tensor)`                       |
| `activation::hard_sigmoid(tensor, alpha, beta)`  | `nn.functional.hardsigmoid(tensor)`                |
| `activation::leaky_relu(tensor, negative_slope)` | `nn.functional.leaky_relu(tensor, negative_slope)` |
| `activation::log_sigmoid(tensor)`                | `nn.functional.log_sigmoid(tensor)`                |
| `activation::log_softmax(tensor, dim)`           | `nn.functional.log_softmax(tensor, dim)`           |
| `activation::mish(tensor)`                       | `nn.functional.mish(tensor)`                       |
| `activation::prelu(tensor,alpha)`                | `nn.functional.prelu(tensor,weight)`               |
| `activation::quiet_softmax(tensor, dim)`         | `nn.functional.quiet_softmax(tensor, dim)`         |
| `activation::relu(tensor)`                       | `nn.functional.relu(tensor)`                       |
| `activation::sigmoid(tensor)`                    | `nn.functional.sigmoid(tensor)`                    |
| `activation::silu(tensor)`                       | `nn.functional.silu(tensor)`                       |
| `activation::softmax(tensor, dim)`               | `nn.functional.softmax(tensor, dim)`               |
| `activation::softmin(tensor, dim)`               | `nn.functional.softmin(tensor, dim)`               |
| `activation::softplus(tensor, beta)`             | `nn.functional.softplus(tensor, beta)`             |
| `activation::tanh(tensor)`                       | `nn.functional.tanh(tensor)`                       |

## Displaying Tensor Details

Burn provides flexible options for displaying tensor information, allowing you to control the level
of detail and formatting to suit your needs.

### Basic Display

To display a detailed view of a tensor, you can simply use Rust's `println!` or `format!` macros:

```rust
let tensor = Tensor::<Backend, 2>::full([2, 3], 0.123456789, &Default::default());
println!("{}", tensor);
```

This will output:

```
Tensor {
  data:
[[0.12345679, 0.12345679, 0.12345679],
 [0.12345679, 0.12345679, 0.12345679]],
  shape:  [2, 3],
  device:  Cpu,
  backend:  "ndarray",
  kind:  "Float",
  dtype:  "f32",
}
```

### Controlling Precision

You can control the number of decimal places displayed using Rust's formatting syntax:

```rust
println!("{:.2}", tensor);
```

Output:

```
Tensor {
  data:
[[0.12, 0.12, 0.12],
 [0.12, 0.12, 0.12]],
  shape:  [2, 3],
  device:  Cpu,
  backend:  "ndarray",
  kind:  "Float",
  dtype:  "f32",
}
```

### Global Print Options

For more fine-grained control over tensor printing, Burn provides a `PrintOptions` struct and a
`set_print_options` function:

```rust
use burn::tensor::{set_print_options, PrintOptions};

let print_options = PrintOptions {
    precision: Some(2),
    ..Default::default()
};

set_print_options(print_options);
```

Options:

- `precision`: Number of decimal places for floating-point numbers (default: None)
- `threshold`: Maximum number of elements to display before summarizing (default: 1000)
- `edge_items`: Number of items to show at the beginning and end of each dimension when summarizing
  (default: 3)

  ### Checking Tensor Closeness

  Burn provides a utility function `check_closeness` to compare two tensors and assess their
  similarity. This function is particularly useful for debugging and validating tensor operations,
  especially when working with floating-point arithmetic where small numerical differences can
  accumulate. It's also valuable when comparing model outputs during the process of importing models
  from other frameworks, helping to ensure that the imported model produces results consistent with
  the original.

  Here's an example of how to use `check_closeness`:

  ```rust
  use burn::tensor::{check_closeness, Tensor};
  type B = burn::backend::NdArray;

  let device = Default::default();
  let tensor1 = Tensor::<B, 1>::from_floats(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.001, 7.002, 8.003, 9.004, 10.1],
      &device,
  );
  let tensor2 = Tensor::<B, 1>::from_floats(
      [1.0, 2.0, 3.0, 4.000, 5.0, 6.0, 7.001, 8.002, 9.003, 10.004],
      &device,
  );

  check_closeness(&tensor1, &tensor2);
  ```

  The `check_closeness` function compares the two input tensors element-wise, checking their
  absolute differences against a range of epsilon values. It then prints a detailed report showing
  the percentage of elements that are within each tolerance level.

  The output provides a breakdown for different epsilon values, allowing you to assess the closeness
  of the tensors at various precision levels. This is particularly helpful when dealing with
  operations that may introduce small numerical discrepancies.

  The function uses color-coded output to highlight the results:

  - Green [PASS]: All elements are within the specified tolerance.
  - Yellow [WARN]: Most elements (90% or more) are within tolerance.
  - Red [FAIL]: Significant differences are detected.

  This utility can be invaluable when implementing or debugging tensor operations, especially those
  involving complex mathematical computations or when porting algorithms from other frameworks. It's
  also an essential tool when verifying the accuracy of imported models, ensuring that the Burn
  implementation produces results that closely match those of the original model.
