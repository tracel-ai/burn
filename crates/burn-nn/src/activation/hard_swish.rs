use burn_core as burn;

use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::activation::hard_swish;
use burn::tensor::backend::Backend;

/// Hard Swish layer.
#[derive(Module, Clone, Debug, Default)]
pub struct HardSwish;

impl HardSwish {
    /// Create the module.
    pub fn new() -> Self {
        Self
    }

    /// Forward pass for the Hard Swish layer.
    ///
    /// See [hard_swish](burn::tensor::activation::hard_swish) for more information.
    ///
    /// # Shapes
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        hard_swish(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::TensorData;
    use burn::tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_hard_swish_forward() {
        let device = <TestBackend as Backend>::Device::default();
        let model = HardSwish::new();

        let input = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[3.0f32, -3.0], [0.0, 1.0]]),
            &device,
        );
        let out = model.forward(input);
        let expected = TensorData::from([[3.0f32, 0.0], [0.0, 0.6666667]]);
        out.to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn display() {
        let layer = HardSwish::new();
        assert_eq!(alloc::format!("{layer}"), "HardSwish");
    }
}
