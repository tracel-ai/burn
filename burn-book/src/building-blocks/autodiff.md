# Autodiff

Burn's tensor also supports auto-differentiation, which is an essential part of any deep learning
framework. We introduced the `Backend` trait in the [previous section](./backend.md), but Burn also
has another trait for autodiff: `AutodiffBackend`.

However, not all tensors support auto-differentiation; you need a backend that implements both the
`Backend` and `AutodiffBackend` traits. Fortunately, you can add auto-differentiation capabilities to any
backend using a backend decorator: `type MyAutodiffBackend = Autodiff<MyBackend>`. This
decorator implements both the `AutodiffBackend` and `Backend` traits by maintaining a dynamic
computational graph and utilizing the inner backend to execute tensor operations.

The `AutodiffBackend` trait adds new operations on float tensors that can't be called otherwise. It also
provides a new associated type, `B::Gradients`, where each calculated gradient resides.

```rust, ignore
fn calculate_gradients<B: AutodiffBackend>(tensor: Tensor<B, 2>) -> B::Gradients {
    let mut gradients = tensor.clone().backward();

    let tensor_grad = tensor.grad(&gradients);        // get
    let tensor_grad = tensor.grad_remove(&mut gradients); // pop

    gradients
}
```

Note that some functions will always be available even if the backend doesn't implement the
`AutodiffBackend` trait. In such cases, those functions will do nothing.

| Burn API                                | PyTorch Equivalent            |
| --------------------------------------- | ----------------------------- |
| `tensor.detach()`                       | `tensor.detach()`             |
| `tensor.require_grad()`                 | `tensor.requires_grad()`      |
| `tensor.is_require_grad()`              | `tensor.requires_grad`        |
| `tensor.set_require_grad(require_grad)` | `tensor.requires_grad(False)` |

However, you're unlikely to make any mistakes since you can't call `backward` on a tensor that is on
a backend that doesn't implement `AutodiffBackend`. Additionally, you can't retrieve the gradient of a
tensor without an autodiff backend.

## Difference with PyTorch

The way Burn handles gradients is different from PyTorch. First, when calling `backward`, each
parameter doesn't have its `grad` field updated. Instead, the backward pass returns all the
calculated gradients in a container. This approach offers numerous benefits, such as the ability to
easily send gradients to other threads.

You can also retrieve the gradient for a specific parameter using the `grad` method on a tensor.
Since this method takes the gradients as input, it's hard to forget to call `backward` beforehand.
Note that sometimes, using `grad_remove` can improve performance by allowing inplace operations.

In PyTorch, when you don't need gradients for inference or validation, you typically need to scope
your code using a block.

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

With Burn, you don't need to wrap the backend with the `Autodiff` for inference, and you
can call `inner()` to obtain the inner tensor, which is useful for validation.

```rust, ignore
/// Use `B: AutodiffBackend`
fn example_validation<B: AutodiffBackend>(tensor: Tensor<B, 2>) {
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

We've seen how gradients can be used with tensors, but the process is a bit different when working
with optimizers from `burn-core`. To work with the `Module` trait, a translation step is required to
link tensor parameters with their gradients. This step is necessary to easily support gradient
accumulation and training on multiple devices, where each module can be forked and run on different
devices in parallel. We'll explore deeper into this topic in the [Module](./module.md) section.

## Retaining the Computational Graph

By default, calling `backward()` consumes the computational graph. Every node is removed from the
graph during the backward pass, freeing activation memory. This is the correct default for
standard training loops where you never need the graph twice.

However, for scientific machine learning use cases such as computing Jacobians, Hessians or
higher-order derivatives, you may need to run multiple backward passes over the same
graph. Burn provides `backward_retain()` for this purpose, equivalent to PyTorch's
`loss.backward(retain_graph=True)`.

```rust, ignore
fn calculate_jacobian<B: AutodiffBackend>(tensor: Tensor<B, 2>) {
    let result = tensor.clone().matmul(tensor.clone());

    // Each slice can be differentiated independently over the same graph
    let result1 = result.clone().slice([0..1, 0..1]).sum();
    let result2 = result.clone().slice([1..2, 0..1]).sum();

    let grads1 = result1.backward_retain(); // graph retained
    let grads2 = result2.backward_retain(); // graph retained again

    let grad1 = tensor.grad(&grads1).unwrap();
    let grad2 = tensor.grad(&grads2).unwrap();
}
```

The tradeoff is memory: `backward_retain()` holds all activation memory for the full duration of
the retained backward passes, whereas standard `backward()` consumes activations as each node
is visited. Only use `backward_retain()` when multiple backward passes over the same graph are
genuinely required.

| Burn API                  | PyTorch Equivalent                        |
| ------------------------- | ----------------------------------------- |
| `tensor.backward()`       | `loss.backward()`                         |
| `tensor.backward_retain()`| `loss.backward(retain_graph=True)`        |