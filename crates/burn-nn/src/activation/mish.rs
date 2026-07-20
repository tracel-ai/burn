use burn_core as burn;

use burn::module::Module;
use burn::tensor::Tensor;

/// Applies the Mish activation function element-wise.
///
/// See also [mish](burn::tensor::activation::mish).
#[derive(Module, Debug, Default)]
pub struct Mish;

impl Mish {
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
        burn::tensor::activation::mish(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Mish::new();

        assert_eq!(alloc::format!("{layer}"), "Mish");
    }
}
