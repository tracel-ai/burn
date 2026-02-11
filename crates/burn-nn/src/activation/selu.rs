use burn_core as burn;

use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Applies the Scaled Exponential Linear Unit function element-wise.
/// See also [selu](burn::tensor::activation::selu)
#[derive(Module, Clone, Debug, Default)]
pub struct Selu;

impl Selu {
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
        burn::tensor::activation::selu(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Selu::new();

        assert_eq!(alloc::format!("{layer}"), "Selu");
    }
}
