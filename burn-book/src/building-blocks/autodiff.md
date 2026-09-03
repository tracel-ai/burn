# Autodiff

Burn tensors support automatic differentiation, which is essential for training neural networks.
Autodiff is selected at runtime by creating an autodiff-enabled device rather than by adding an
`Autodiff<B>` type parameter to tensors and modules.

```rust, ignore
use burn::tensor::{Device, Tensor};

let device = Device::wgpu(Default::default()).autodiff();
let tensor = Tensor::<2>::ones([2, 2], &device).require_grad();
let output = tensor.clone().powf_scalar(2.0).sum();
let mut gradients = output.backward();

let tensor_grad = tensor.grad(&gradients);             // get
let tensor_grad = tensor.grad_remove(&mut gradients);  // pop
```

Calling `backward` returns the calculated gradients in a container instead of updating a `grad`
field on every parameter. Passing that container to `grad` or `grad_remove` makes the relationship
between the backward pass and gradient access explicit. `grad_remove` can also enable in-place
optimizations when a gradient is consumed only once.

Note that some functions will always be available even if the tensor is not on an autodiff-enabled
device. In such cases, those functions will do nothing.

| Burn API                                | PyTorch Equivalent           |
| --------------------------------------- | ---------------------------- |
| `tensor.detach()`                       | `tensor.detach()`            |
| `tensor.require_grad()`                 | `tensor.requires_grad_()`    |
| `tensor.is_require_grad()`              | `tensor.requires_grad`       |
| `tensor.set_require_grad(require_grad)` | `tensor.requires_grad_(...)` |

## Difference with PyTorch

The way Burn handles gradients is different from PyTorch. First, when calling `backward`, each
parameter doesn't have its `grad` field updated. Instead, the backward pass returns all the
calculated gradients in a container. This approach offers numerous benefits, such as the ability to
easily send gradients to other threads.

In PyTorch, when you don't need gradients for inference or validation, you typically need to scope
your code using a block.

```python
# Inference mode
torch.inference_mode():
   # your code
   ...

# Or no grad
torch.no_grad():
   # your code
   ...
```

With Burn, tensors shouldn't be on an autodiff device for inference, and you can call
`without_autodiff()` to obtain a tensor without autodiff, which is useful for validation. The
historical `inner()` method is equivalent.

```rust, ignore
fn example_validation(tensor: Tensor<2>) {
    debug_assert!(tensor.device().is_autodiff());
    let inner_tensor = tensor.without_autodiff();
    let _ = inner_tensor + 5;
}

fn example_inference(tensor: Tensor<2>) {
    debug_assert!(!tensor.device().is_autodiff());
    let _ = tensor + 5;
    ...
}
```

## Gradients with Optimizers

We've seen how gradients can be used with tensors, but the process is a bit different when working
with optimizers from `burn-optim`. To work with the `Module` trait, a translation step is required to
link tensor parameters with their gradients. This step is necessary to easily support gradient
accumulation and training on multiple devices, where each module can be forked and run on different
devices in parallel. The [Optimizer](./optimizer.md) section explains how those gradients update
module parameters.
