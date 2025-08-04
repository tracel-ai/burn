use crate as burn;
use crate::module::Module;
use crate::tensor::Tensor;
use crate::tensor::backend::Backend;

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
    /// * The input tensor along `dim` must have an even size along the specified dimension. N is divisible by 2.
    /// * Negative indices for `dim` are not supported (unlike PyTorch's nn.GLU).
    /// 
    /// ### Arguments
    /// * `tensor` - The input tensor. With shape `[∗1,N,∗2]` where `*` means, any number of additional dimensions
    ///
    /// ### Returns
    /// * Output tensor with shape `[∗1​,M,∗2​]` where M=N/2
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        crate::tensor::activation::glu(input, self.dim)
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
