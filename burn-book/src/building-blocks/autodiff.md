# Autodiff

Burn's tensor also supports autodifferentiation, which is an essential part of any deep learning framework.
We introduced the `Backend` trait in the [previous section](./backend.md), but Burn also has another trait for autodiff: `ADBackend`.

However, not all tensors support auto-differentiation; you need a backend that implements both the `Backend` and `ADBackend` traits.
Fortunately, you can add autodifferentiation capabilities to any backend using a backend decorator: `type MyAutodiffBackend = ADBackendDecorator<MyBackend>`.
This decorator implements both the `ADBackend` and `Backend` traits by maintaining a dynamic computational graph and utilizing the inner backend to execute tensor operations.

The `ADBackend` trait adds new operations on float tensors that can't be called otherwise.
It also provides a new associated type, `B::Gradients`, where each calculated gradient resides.

```rust, ignore
fn calculate_gradients<B: ADBackend>(tensor: Tensor<B, 2>) -> B::Gradients {
    let mut gradients = tensor.clone().backward();

    let tensor_grad = tensor.grad(&gradients);        // get
    let tensor_grad = tensor.grad_remove(&mut gradients); // pop

    gradients
}
```

Note that some functions will always be available even if the backend doesn't implement the `ADBackend` trait.
In such cases, those functions will do nothing.

| Burn API                                               | PyTorch Equivalent                                   |
|--------------------------------------------------------|------------------------------------------------------|
| `tensor.detach()`                                      | `tensor.detach()`                                    |
| `tensor.require_grad()`                                | `tensor.requires_grad()`                             |
| `tensor.is_require_grad()`                             | `tensor.requires_grad`                               |
| `tensor.set_require_grad(require_grad)`                | `tensor.requires_grad(False)`                        |

However, you're unlikely to make any mistakes since you can't call `backward` on a tensor that is on a backend that doesn't implement `ADBackend`.
Additionally, you can't retrieve the gradient of a tensor without an autodiff backend.

## Difference with PyTorch

The way Burn handles gradients is different from PyTorch.
First, when calling `backward`, each parameter doesn't have its `grad` field updated.
Instead, the backward pass returns all the calculated gradients in a container.
This approach offers numerous benefits, such as the ability to easily send gradients to other threads.

You can also retrieve the gradient for a specific parameter using the `grad` method on a tensor.
Since this method takes the gradients as input, it's hard to forget to call `backward` beforehand.
Note that sometimes, using `grad_remove` can improve performance by allowing inplace operations.

In PyTorch, when you don't need gradients for inference or validation, you typically need to scope your code using a block.

```python
# Inference mode
torch.inference():
   # your code
   ...

# Or no grad
torch.no_grad():
   # your code
   ...
```

With Burn, you don't need to wrap the backend with the `ADBackendDecorator` for inference, and you can call `inner()` to obtain the inner tensor, which is useful for validation.

```rust, ignore
/// Use `B: ADBackend`
fn example_validation<B: ADBackend>(tensor: Tensor<B, 2>) {
    let inner_tensor: Tensor<B::InnerBackend, 2> = tensor.inner();
    let _ = inner_tensor + 5;
}

/// Use `B: Backend`
fn example_inference<B: Backend>(tensor: Tensor<B, 2>) {
    let _ = tensor + 5;
    ...
}
```

**Gradients with Optimizers**

We've seen how gradients can be used with tensors, but the process is a bit different when working with optimizers from `burn-core`.
To work with the `Module` trait, a translation step is required to link tensor parameters with their gradients.
This step is necessary to easily support gradient accumulation and training on multiple devices, where each module can be forked and run on different devices in parallel.
We'll explore deeper into this topic in the [Module](./module.md) section.
