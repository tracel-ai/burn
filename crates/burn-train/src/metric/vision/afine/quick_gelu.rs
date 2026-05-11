//! QuickGELU activation used by OpenAI's CLIP.
//!
//! Defined as `x * sigmoid(1.702 * x)`. Distinct from the erf-based
//! [`burn_nn::Gelu`]: CLIP was trained with QuickGELU and substituting
//! standard GELU silently shifts every transformer-block MLP output.

use burn_core as burn;

use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;

/// Approximation coefficient. Must be exactly 1.702; CLIP weights are
/// trained with this value and changing it breaks numerical parity.
const COEFFICIENT: f64 = 1.702;

/// QuickGELU activation: `x * sigmoid(1.702 * x)`.
#[derive(Module, Clone, Debug, Default)]
pub(crate) struct QuickGelu;

impl QuickGelu {
    /// Apply the activation element-wise. Shape-preserving.
    pub(crate) fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let scaled = x.clone().mul_scalar(COEFFICIENT);
        x * sigmoid(scaled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::tensor::{Tolerance, ops::FloatElem};
    use burn_flex::Flex;

    type TestBackend = Flex;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn quick_gelu_matches_formula() {
        let device = Default::default();
        let activation = QuickGelu;

        let input = Tensor::<TestBackend, 2>::from_floats([[-1.0, 0.0, 1.0, 2.0]], &device);
        let output = activation.forward(input);

        // Hand-computed: x * sigmoid(1.702 * x).
        // sigmoid(-1.702) = 1 / (1 + exp(1.702))  ≈ 0.154160
        // sigmoid( 1.702)                          ≈ 0.845840
        // sigmoid( 3.404)                          ≈ 0.967812
        let expected = Tensor::<TestBackend, 2>::from_floats(
            [[-0.154_160, 0.0, 0.845_840, 1.935_624]],
            &device,
        );

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::default());
    }

    #[test]
    fn quick_gelu_preserves_shape() {
        let device = Default::default();
        let activation = QuickGelu;

        let input = Tensor::<TestBackend, 3>::zeros([2, 5, 8], &device);
        let output = activation.forward(input);

        assert_eq!(output.dims(), [2, 5, 8]);
    }
}
