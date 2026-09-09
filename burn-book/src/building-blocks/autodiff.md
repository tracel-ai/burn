# Autodiff

Burn tensors support automatic differentiation, which is essential for training neural networks.
Autodiff is selected at runtime. Devices provide the autodiff and checkpointing defaults for newly
created tensors; each tensor carries its own context and can change it independently. Inspect that
context with `tensor.is_autodiff()`. Moving a tensor to a device does not apply the destination's
autodiff defaults.

The user-facing tensor and module types no longer distinguish `B: Backend` from
`B: AutodiffBackend`; autodiff APIs check their preconditions at runtime. Enabling autodiff permits
graph recording but does not make every input require gradients. Use `require_grad()` on source
leaves whose gradients you need, and let ordinary model inputs remain constants when their
gradients aren't needed. For modules, `train()` enables autodiff and restores configured parameter
trainability and training flags; see [module training state](./module.md#methods).

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

Autodiff association, graph participation, and gradient retention are distinct, but constrained,
properties:

| Property               | Accessor                                   | Related APIs                                           |
| ---------------------- | ------------------------------------------ | ------------------------------------------------------ |
| Autodiff association   | `tensor.is_autodiff()`                     | `autodiff()` / `without_autodiff()`                    |
| Graph participation    | `tensor.is_tracked()`                      | `detach()` / operations with tracked inputs            |
| Gradient retention     | `tensor.is_require_grad()`                 | `require_grad()` / `set_require_grad(...)`             |
| Checkpointing strategy | `tensor.gradient_checkpointing_strategy()` | `autodiff().with_gradient_checkpointing_strategy(...)` |

`require_grad()` makes an autodiff leaf participate in the graph and retain its gradient; it does
not enable autodiff. On a floating-point tensor without autodiff, it panics; call `.autodiff()`
first. On a tracked non-leaf, it also panics:
retaining intermediate gradients while preserving their source graph is currently unsupported.
`set_require_grad(false)` starts a new untracked lineage, cutting any connection to upstream
tensors; it doesn't merely disable gradient storage. Disabling gradients on a plain tensor is
harmless. Quantized tensors cannot retain gradients; their `require_grad()` and
`set_require_grad(...)` calls leave them unchanged.

`detach()` keeps the autodiff association but starts a new graph lineage, preserving a leaf's
gradient-retention setting. `without_autodiff()` removes the association entirely.

`to_device()` preserves the source tensor's autodiff association and checkpointing strategy,
ignoring the destination's autodiff configuration. For tracked inputs, it records a differentiable
operation even when the device is unchanged. Choose the transfer according to where gradients
should flow:

```rust, ignore
let moved = source.clone().to_device(&destination); // Gradients flow back to source.
let leaf = source.to_device(&destination).detach().require_grad(); // New destination leaf.
```

The first result cannot retain its own gradient; retrieve the source's gradient after backward.
The second can retain its gradient, but is disconnected from the source graph. Distributed
backward currently requires every distributed parameter to use the same backend as the loss;
incompatible graphs are rejected before synchronization or gradient computation begins.

When combining tensors that both have autodiff enabled, their checkpointing strategies must match
or the operation panics. A transferred tensor keeps its source strategy, which may differ from
that of tensors newly created on the destination. For example, a transferred `Balanced` tensor
cannot combine with a new autodiff tensor using the destination's `Disabled` strategy. Create the
other operand on `moved.device()` to inherit the matching context, or explicitly align the
operands with `with_gradient_checkpointing_strategy(...)`.

For floating-point tensors, retained gradients imply graph participation, and graph participation
implies an autodiff association:

```text
is_require_grad() => is_tracked() => is_autodiff()
gradient_checkpointing_strategy().is_some() == is_autodiff()
```

## Difference with PyTorch

Similarly named APIs do not always have the same semantics:

- Burn's `is_require_grad()` reports gradient retention. PyTorch's
  [`requires_grad`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.requires_grad.html)
  also applies to tracked intermediate tensors whose gradients are not retained. Burn's
  `is_tracked()` is the closer comparison for graph participation.
- Burn's `detach()` preserves a leaf's gradient-retention setting. PyTorch's
  [`detach()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html) always returns
  a tensor that does not require gradients. Use `set_require_grad(false)` in Burn to start a new
  lineage with gradient retention disabled while keeping the autodiff association.

The way Burn handles gradients is different from PyTorch. First, when calling `backward`, each
parameter doesn't have its `grad` field updated. Instead, the backward pass returns all the
calculated gradients in a container. This approach offers numerous benefits, such as the ability to
easily send gradients to other threads.

In PyTorch, when you don't need gradients for inference or validation, you typically need to scope
your code using a block.

```python
# Inference mode
with torch.inference_mode():
   # your code
   ...

# Or no grad
with torch.no_grad():
   # your code
   ...
```

With Burn, call `without_autodiff()` on a tensor to remove its autodiff association for inference
or validation. Moving it to a device without autodiff leaves its existing association intact.
The historical `inner()` method is equivalent to `without_autodiff()`.

When an operation combines a tensor with autodiff and a tensor without it, the operation uses
autodiff and treats the latter tensor as a constant. The original tensor remains unchanged.

```rust, ignore
fn example_validation(tensor: Tensor<2>) {
    debug_assert!(tensor.is_autodiff());
    let inner_tensor = tensor.without_autodiff();
    let _ = inner_tensor + 5;
}

fn example_inference(tensor: Tensor<2>) {
    debug_assert!(!tensor.is_autodiff());
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
