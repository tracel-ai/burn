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

The following tensor methods control graph tracking:

| Burn API                                | PyTorch Equivalent            |
| --------------------------------------- | ----------------------------- |
| `tensor.detach()`                       | `tensor.detach()`             |
| `tensor.require_grad()`                 | `tensor.requires_grad_(True)` |
| `tensor.is_require_grad()`              | `tensor.requires_grad`        |
| `tensor.set_require_grad(require_grad)` | `tensor.requires_grad_(...)`  |

For inference, use a regular device. For validation during a training workflow, tensors are detached
from the graph:

```rust, ignore
fn validation(tensor: Tensor<2>) {
    let tensor = tensor.detach();
    let _ = tensor + 5;
}

fn inference(tensor: Tensor<2>) {
    let _ = tensor + 5;
}
```

## Gradients with Optimizers

When using optimizers from `burn-core`, the module translates gradients from the tensor gradient
container to its parameter records. This supports gradient accumulation and training on multiple
devices without making the module type depend on a backend. The
[Module](./module.md) section explores modules and parameter mapping in more detail.
