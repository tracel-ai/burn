use burn_core as burn;

use burn::module::Module;
use burn::tensor::Tensor;

/// Applies the LogSigmoid activation function element-wise.
///
/// See also [log_sigmoid](burn::tensor::activation::log_sigmoid).
#[derive(Module, Debug, Default)]
pub struct LogSigmoid;

impl LogSigmoid {
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
        burn::tensor::activation::log_sigmoid(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = LogSigmoid::new();

        assert_eq!(alloc::format!("{layer}"), "LogSigmoid");
    }
}
