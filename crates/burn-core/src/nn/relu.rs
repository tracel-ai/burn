use crate as burn;

use crate::module::Module;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

/// Applies the rectified linear unit function element-wise
/// See also [relu](burn::tensor::activation::relu)
///
#[derive(Module, Clone, Debug, Default)]
pub struct Relu;

impl Relu {
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
        crate::tensor::activation::relu(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Relu::new();

        assert_eq!(alloc::format!("{}", layer), "Relu");
    }
}
