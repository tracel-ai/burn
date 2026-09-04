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

Autodiff association, graph participation, and gradient retention are related but independent
properties:

| Property               | Accessor                                   | Related APIs                                                                             |
| ---------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------- |
| Autodiff association   | `tensor.is_autodiff()`                     | `autodiff()` / `without_autodiff()`                                                      |
| Graph participation    | `tensor.is_tracked()`                      | `detach()` / operations with tracked inputs                                              |
| Gradient retention     | `tensor.is_require_grad()`                 | `require_grad()` / `set_require_grad(...)`                                               |
| Checkpointing strategy | `tensor.gradient_checkpointing_strategy()` | `autodiff().with_gradient_checkpointing_strategy(...)`                                   |

`require_grad()` only controls whether a tensor's gradient is retained; it does not enable autodiff.
On a tensor without autodiff, it is a no-op. `detach()` keeps the autodiff association but starts a
new graph lineage, while `without_autodiff()` removes the association entirely.

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

When an operation combines a tensor with autodiff and a tensor without it, the operation uses
autodiff and treats the latter tensor as a constant. The original tensor remains unchanged.

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
with optimizers from `burn-optim`. To work with the `Module` trait, a translation step is required
to link tensor parameters with their gradients. This step is necessary to easily support gradient
accumulation and training on multiple devices, where each module can be forked and run on different
devices in parallel. The [Optimizer](./optimizer.md) section explains how those gradients update
module parameters.
