# Modules

The `Module` derive allows you to create your own neural network modules, similar to PyTorch. The
derive function only generates the necessary methods to essentially act as a parameter container for
your type, it makes no assumptions about how the forward pass is declared.

```rust
use burn::nn;
use burn::module::Module;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct PositionWiseFeedForward<B: Backend> {
    linear_inner: Linear<B>,
    linear_outer: Linear<B>,
    dropout: Dropout,
    gelu: GELU,
}

impl<B: Backend> PositionWiseFeedForward<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear_inner.forward(input);
        let x = self.gelu.forward(x);
        let x = self.dropout.forward(x);

        self.linear_outer.forward(x)
    }
}
```

Note that all fields declared in the struct must also implement the `Module` trait. The `Tensor`
struct doesn't implement `Module`, but `Param<Tensor<B, D>>` does.