use burn_core as burn;

use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Applies the approximate Gaussian Error Linear Units function element-wise
/// using the tanh approximation (matches HuggingFace `gelu_new`).
///
/// `GeluApproximate(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
#[derive(Module, Clone, Debug, Default)]
pub struct GeluApproximate;

impl GeluApproximate {
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
        // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let x = input;
        let inner = x.clone() + x.clone().powf_scalar(3.0) * 0.044715;
        let inner = inner * core::f64::consts::FRAC_2_PI.sqrt();
        (x.clone() * (inner.tanh() + 1)) * 0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::Tolerance;
    use burn::tensor::ops::FloatElem;

    type FT = FloatElem<TestBackend>;

    #[test]
    fn display() {
        let layer = GeluApproximate::new();

        assert_eq!(alloc::format!("{layer}"), "GeluApproximate");
    }

    #[test]
    fn forward() {
        let device = Default::default();
        // Values: negative, zero, positive
        let input =
            Tensor::<TestBackend, 2>::from_data([[-1.0, 0.0, 1.0], [0.5, -0.5, 2.0]], &device);

        let output = GeluApproximate.forward(input);

        // PyTorch: torch.nn.functional.gelu(x, approximate="tanh")
        let expected = Tensor::<TestBackend, 2>::from_data(
            [
                [-0.1588079929, 0.0000000000, 0.8411920071],
                [0.3457140028, -0.1542859972, 1.9545977116],
            ],
            &device,
        );

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::rel_abs(1e-5, 1e-5));
    }
}
