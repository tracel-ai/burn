use crate as burn;
use crate::config::Config;
use crate::tensor::backend::Backend;
use crate::tensor::{Distribution, Tensor};

/// Configuration to create a [Dropout](Dropout) layer.
#[derive(Config)]
pub struct DropoutConfig {
    /// The probability of randomly zeroes some elements of the input tensor during training.
    pub prob: f64,
}

/// Set at random some elements of the input tensor to zero during training.
///
/// This is an effective regularization technique as describe in the paper
/// [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580).
///
/// The input is also scaled during training to `1 / (1-p)`.
#[derive(Clone, Debug)]
pub struct Dropout {
    prob: f64,
}

impl Dropout {
    /// Create the module from the given configuration.
    pub fn new(config: &DropoutConfig) -> Self {
        Self { prob: config.prob }
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        if !B::ad_enabled() || self.prob == 0.0 {
            return input;
        }

        let random = input.random_like(Distribution::Bernoulli(self.prob));
        let mask = random.equal_scalar(1);
        let x = input.mask_fill(&mask, 0.0_f32);

        x / (1.0 - self.prob)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;
    use crate::{TestADBackend, TestBackend};

    #[test]
    fn with_ad_backend_should_mark_input() {
        let tensor = Tensor::<TestADBackend, 2>::ones(Shape::new([100, 100]));
        let dropout = Dropout::new(&DropoutConfig { prob: 0.5 });

        let output = dropout.forward(tensor.clone());

        assert_ne!(tensor.to_data(), output.to_data());
    }

    #[test]
    fn without_ad_backend_should_not_change_input() {
        let tensor = Tensor::<TestBackend, 2>::ones(Shape::new([100, 100]));
        let dropout = Dropout::new(&DropoutConfig { prob: 0.5 });

        let output = dropout.forward(tensor.clone());

        assert_eq!(tensor.to_data(), output.to_data());
    }
}
