use burn_core as burn;

use burn::module::Module;
use burn::tensor::Tensor;

/// Applies the Tanhshrink function element-wise, `x - tanh(x)`.
/// See also [tanhshrink](burn::tensor::activation::tanhshrink)
///
#[derive(Module, Debug, Default)]
pub struct Tanhshrink;

impl Tanhshrink {
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
        burn::tensor::activation::tanhshrink(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Tanhshrink::new();

        assert_eq!(alloc::format!("{layer}"), "Tanhshrink");
    }
}
