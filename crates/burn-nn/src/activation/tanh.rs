use burn_core as burn;

use burn_core::module::Module;
use burn_tensor::Tensor;
use burn_tensor::backend::Backend;

/// Applies the tanh activation function element-wise
/// See also [tanh](burn::tensor::activation::tanh)
#[derive(Module, Clone, Debug, Default)]
pub struct Tanh;

impl Tanh {
    /// Create the module.
    pub fn new() -> Self {
        Self {}
    }
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        burn_tensor::activation::tanh(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Tanh::new();

        assert_eq!(alloc::format!("{layer}"), "Tanh");
    }
}
