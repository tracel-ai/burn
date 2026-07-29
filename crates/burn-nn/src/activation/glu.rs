use burn_core as burn;

use burn::module::Module;
use burn::tensor::{AsIndex, Tensor};

/// Applies the gated linear unit function.
///
/// Negative dimensions are supported and count from the end.
///
/// See also [glu](burn::tensor::activation::glu)
#[derive(Module, Debug, Default)]
pub struct GLU {
    dim: isize,
}

impl GLU {
    /// Create the module.
    ///
    /// # Arguments
    /// * `dim` - The dimension on which to split the input.
    pub fn new(dim: impl AsIndex) -> Self {
        Self {
            dim: dim.as_index(),
        }
    }

    /// Applies the gated linear unit function.
    ///
    /// GLU(a,b)=a⊗σ(b) where `a` is the first half of the input matrices and `b` is the second half.
    ///
    /// **Note**:
    /// * The size of the input tensor along `dim` must be divisible by 2.
    /// * Negative dimensions are supported and count from the end.
    ///
    /// ### Arguments
    /// * `tensor` - The input tensor.
    ///
    /// ### Returns
    /// * A tensor with the same shape as the input, except the size along `dim` is halved.
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        burn::tensor::activation::glu(input, self.dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{TensorData, Tolerance};

    type FT = f32;

    #[test]
    fn display() {
        let layer = GLU::new(1);

        assert_eq!(alloc::format!("{layer}"), "GLU {\n  dim: 1\n}");
    }

    #[test]
    fn forward_negative_dim() {
        let device = Default::default();
        let tensor = Tensor::<3>::from_data(
            [[
                [
                    -0.5710_f32,
                    -1.3416,
                    1.9128,
                    -0.8257,
                    -0.1331,
                    -1.4804,
                    -0.6281,
                    -0.6115,
                ],
                [
                    0.0267, -1.3834, 0.2752, 0.7844, -0.3549, -0.4274, 0.3290, -0.5459,
                ],
                [
                    -1.6347, -2.0908, 1.8801, 0.3541, 0.2237, 1.0377, 2.4850, 0.3490,
                ],
            ]],
            &device,
        );

        let output = GLU::new(-1).forward(tensor);

        output.into_data().assert_approx_eq::<FT>(
            &TensorData::from([[
                [-0.2665, -0.2487, 0.6656, -0.2904],
                [0.0110, -0.5461, 0.1601, 0.2877],
                [-0.9084, -1.5439, 1.7355, 0.2077],
            ]]),
            Tolerance::default(),
        );
    }
}
