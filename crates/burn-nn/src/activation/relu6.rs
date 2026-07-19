use burn_core as burn;

use burn::module::Module;
use burn::tensor::Tensor;

/// Applies the ReLU6 function element-wise, clamping the output to `[0, 6]`.
/// See also [relu6](burn::tensor::activation::relu6)
///
#[derive(Module, Debug, Default)]
pub struct Relu6;

impl Relu6 {
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
        burn::tensor::activation::relu6(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Relu6::new();

        assert_eq!(alloc::format!("{layer}"), "Relu6");
    }
}
