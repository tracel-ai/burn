use burn_core as burn;

use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Applies the gated linear unit function.
///
/// See also [glu](burn::tensor::activation::glu)
#[derive(Module, Clone, Debug, Default)]
pub struct GLU {
    dim: usize,
}

impl GLU {
    /// Create the module.
    ///
    /// # Arguments
    /// * `dim` - The dimension on which to split the input.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Applies the gated linear unit function.
    ///
    /// GLU(a,b)=a⊗σ(b) where `a` is the first half of the input matrices and `b` is the second half.
    ///
    /// **Note**:
    /// * The size of the input tensor along `dim` must be divisible by 2.
    ///
    /// ### Arguments
    /// * `tensor` - The input tensor.
    ///
    /// ### Returns
    /// * A tensor with the same shape as the input, except the size along `dim` is halved.
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        burn::tensor::activation::glu(input, self.dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = GLU::new(1);

        assert_eq!(alloc::format!("{layer}"), "GLU {\n  dim: 1\n}");
    }
}
