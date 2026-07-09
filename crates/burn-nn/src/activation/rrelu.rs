use burn::config::Config;
use burn::module::Module;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::{Distribution, Tensor};
use burn_core as burn;

use burn::tensor::activation::leaky_relu;

/// Randomized Leaky ReLU (RReLU) layer, from
/// [Empirical Evaluation of Rectified Activations](https://arxiv.org/abs/1505.00853).
///
/// During training the negative slope is sampled element-wise from a uniform
/// distribution `[lower, upper)`; during evaluation the fixed midpoint slope
/// `(lower + upper) / 2` is used, which is identical to a LeakyReLU. Following
/// the same convention as [Dropout](crate::Dropout), the training behaviour is
/// enabled when the input is on an autodiff backend.
///
/// Should be created with [RReluConfig](RReluConfig).
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct RRelu {
    /// The lower bound of the uniform slope range.
    pub lower: f64,
    /// The upper bound of the uniform slope range.
    pub upper: f64,
}

/// Configuration to create a [RRelu](RRelu) layer using the [init function](RReluConfig::init).
#[derive(Config, Debug)]
pub struct RReluConfig {
    /// The lower bound of the uniform slope range. Default: 1/8 = 0.125.
    #[config(default = "0.125")]
    pub lower: f64,
    /// The upper bound of the uniform slope range. Default: 1/3.
    #[config(default = "1.0 / 3.0")]
    pub upper: f64,
}

impl RReluConfig {
    /// Initialize a new [RRelu](RRelu) layer.
    pub fn init(&self) -> RRelu {
        assert!(
            self.lower <= self.upper,
            "RRelu: lower bound ({}) must be <= upper bound ({})",
            self.lower,
            self.upper
        );
        RRelu {
            lower: self.lower,
            upper: self.upper,
        }
    }
}

impl ModuleDisplay for RRelu {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("lower", &self.lower)
            .add("upper", &self.upper)
            .optional()
    }
}

impl RRelu {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [RRelu](RRelu) for more information.
    ///
    /// # Shapes
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        if !input.device().is_autodiff() {
            // Evaluation: fixed midpoint slope (identical to LeakyReLU).
            return leaky_relu(input, (self.lower + self.upper) / 2.0);
        }

        // Training: sample a per-element slope in [lower, upper) and apply it to
        // the negative part only (positives pass through unchanged).
        let is_negative = input.clone().lower_elem(0.0);
        let slope = input.random_like(Distribution::Uniform(self.lower, self.upper));
        let scaled = input.clone() * slope;
        input.mask_where(is_negative, scaled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::Tolerance;
    type FT = f32;

    #[test]
    fn eval_matches_leaky_relu_midpoint() {
        // On a non-autodiff device, RReLU is deterministic: leaky_relu(x, 0.2).
        let device = Default::default();
        let model = RReluConfig::new().with_lower(0.1).with_upper(0.3).init();

        let input =
            Tensor::<2>::from_data(TensorData::from([[-2.0, -1.0, 0.0], [0.5, 1.0, 2.0]]), &device);
        let output = model.forward(input);

        // midpoint slope = (0.1 + 0.3) / 2 = 0.2
        let expected = TensorData::from([[-0.4, -0.2, 0.0], [0.5, 1.0, 2.0]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[cfg(feature = "std")]
    #[test]
    fn training_scales_negatives_randomly() {
        use burn::tensor::Device;
        // On an autodiff device the negative slopes are randomised, so the
        // output must differ from the input for a strictly-negative tensor.
        let device = Device::default().autodiff();
        let model = RReluConfig::new().init();

        let input = Tensor::<2>::from_data(TensorData::from([[-1.0, -2.0], [-3.0, -4.0]]), &device);
        let output = model.forward(input.clone());

        assert_ne!(input.to_data(), output.to_data());
    }

    #[test]
    fn display() {
        let layer = RReluConfig::new().with_lower(0.1).with_upper(0.3).init();
        assert_eq!(alloc::format!("{layer}"), "RRelu {lower: 0.1, upper: 0.3}");
    }

    #[test]
    #[should_panic = "must be <= upper bound"]
    fn rejects_lower_above_upper() {
        let _ = RReluConfig::new().with_lower(0.5).with_upper(0.2).init();
    }
}
