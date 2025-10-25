use burn_core as burn;

use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Applies the Gaussian Error Linear Units function element-wise.
/// See also [gelu](burn::tensor::activation::gelu)
#[derive(Module, Clone, Debug, Default)]
pub struct Gelu;

impl Gelu {
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
        burn::tensor::activation::gelu(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Gelu::new();

        assert_eq!(alloc::format!("{layer}"), "Gelu");
    }
}
