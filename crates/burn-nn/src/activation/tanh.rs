use burn_core as burn;

use burn::module::Module;
use burn::tensor::Tensor;

/// Applies the tanh activation function element-wise
/// See also [tanh](burn::tensor::activation::tanh)
#[derive(Module, Debug, Default)]
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
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        burn::tensor::activation::tanh(input)
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
