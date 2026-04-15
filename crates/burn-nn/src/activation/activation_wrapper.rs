use burn_core as burn;

use crate::activation::{
    Celu, CeluConfig, Elu, EluConfig, Gelu, HardShrink, HardShrinkConfig, HardSigmoid,
    HardSigmoidConfig, HardSwish, LeakyRelu, LeakyReluConfig, PRelu, PReluConfig, Relu, Selu,
    Shrink, ShrinkConfig, Sigmoid, SoftShrink, SoftShrinkConfig, Softplus, SoftplusConfig,
    Softsign, SwiGlu, SwiGluConfig, Tanh, ThresholdedRelu, ThresholdedReluConfig,
};
use burn::config::Config;
use burn::module::Module;
use burn::tensor::Device;
use burn::tensor::Tensor;

/// [`Activation`] Configuration.
#[derive(Config, Debug)]
#[non_exhaustive]
pub enum ActivationConfig {
    /// [`Gelu`] activation layer.
    Gelu,

    /// [`Gelu`] activation layer with tanh approximation.
    GeluApproximate,

    /// [`PRelu`] activation layer.
    PRelu(PReluConfig),

    /// [`Relu`] activation layer.
    Relu,

    /// [`LeakyRelu`] activation layer.
    LeakyRelu(LeakyReluConfig),

    /// [`SwiGlu`] activation layer.
    SwiGlu(SwiGluConfig),

    /// [`Selu`] activation layer.
    Selu,

    /// [`Sigmoid`] activation layer.
    Sigmoid,

    /// [`Tanh`] activation layer.
    Tanh,

    /// [`HardSigmoid`] activation layer.
    HardSigmoid(HardSigmoidConfig),

    /// [`HardSwish`] activation layer.
    HardSwish,

    /// [`Softplus`] activation layer.
    Softplus(SoftplusConfig),

    /// [`Softsign`] activation layer.
    Softsign,

    /// [`Elu`] activation layer.
    Elu(EluConfig),

    /// [`Celu`] activation layer.
    Celu(CeluConfig),

    /// [`ThresholdedRelu`] activation layer.
    ThresholdedRelu(ThresholdedReluConfig),

    /// [`HardShrink`] activation layer.
    HardShrink(HardShrinkConfig),

    /// [`SoftShrink`] activation layer.
    SoftShrink(SoftShrinkConfig),

    /// [`Shrink`] activation layer.
    Shrink(ShrinkConfig),
}

impl From<PReluConfig> for ActivationConfig {
    fn from(config: PReluConfig) -> Self {
        Self::PRelu(config)
    }
}

impl From<LeakyReluConfig> for ActivationConfig {
    fn from(config: LeakyReluConfig) -> Self {
        Self::LeakyRelu(config)
    }
}

impl From<SwiGluConfig> for ActivationConfig {
    fn from(config: SwiGluConfig) -> Self {
        Self::SwiGlu(config)
    }
}

impl From<HardSigmoidConfig> for ActivationConfig {
    fn from(config: HardSigmoidConfig) -> Self {
        Self::HardSigmoid(config)
    }
}

impl From<SoftplusConfig> for ActivationConfig {
    fn from(config: SoftplusConfig) -> Self {
        Self::Softplus(config)
    }
}

impl From<EluConfig> for ActivationConfig {
    fn from(config: EluConfig) -> Self {
        Self::Elu(config)
    }
}

impl From<CeluConfig> for ActivationConfig {
    fn from(config: CeluConfig) -> Self {
        Self::Celu(config)
    }
}

impl From<ThresholdedReluConfig> for ActivationConfig {
    fn from(config: ThresholdedReluConfig) -> Self {
        Self::ThresholdedRelu(config)
    }
}

impl From<HardShrinkConfig> for ActivationConfig {
    fn from(config: HardShrinkConfig) -> Self {
        Self::HardShrink(config)
    }
}

impl From<SoftShrinkConfig> for ActivationConfig {
    fn from(config: SoftShrinkConfig) -> Self {
        Self::SoftShrink(config)
    }
}

impl From<ShrinkConfig> for ActivationConfig {
    fn from(config: ShrinkConfig) -> Self {
        Self::Shrink(config)
    }
}

impl ActivationConfig {
    /// Initialize a wrapped activation layer.
    pub fn init(&self, device: &Device) -> Activation {
        match self {
            ActivationConfig::Relu => Relu.into(),
            ActivationConfig::LeakyRelu(conf) => conf.init().into(),
            ActivationConfig::Gelu => Gelu::new().into(),
            ActivationConfig::GeluApproximate => Gelu::new_approximate().into(),
            ActivationConfig::PRelu(conf) => conf.init(device).into(),
            ActivationConfig::SwiGlu(conf) => conf.init(device).into(),
            ActivationConfig::HardSigmoid(conf) => conf.init().into(),
            ActivationConfig::HardSwish => HardSwish.into(),
            ActivationConfig::Softplus(conf) => conf.init().into(),
            ActivationConfig::Selu => Selu.into(),
            ActivationConfig::Sigmoid => Sigmoid.into(),
            ActivationConfig::Tanh => Tanh.into(),
            ActivationConfig::Softsign => Softsign.into(),
            ActivationConfig::Elu(conf) => conf.init().into(),
            ActivationConfig::Celu(conf) => conf.init().into(),
            ActivationConfig::HardShrink(conf) => conf.init().into(),
            ActivationConfig::SoftShrink(conf) => conf.init().into(),
            ActivationConfig::Shrink(conf) => conf.init().into(),
            ActivationConfig::ThresholdedRelu(conf) => conf.init().into(),
        }
    }
}

/// Activation Layer Wrapper.
///
/// Provides support for many in-built `burn::nn` activations.
#[derive(Module, Debug)]
#[non_exhaustive]
#[allow(clippy::large_enum_variant)]
pub enum Activation {
    /// [`Gelu`] activation layer.
    Gelu(Gelu),

    /// [`PRelu`] activation layer.
    PRelu(PRelu),

    /// [`Relu`] activation layer.
    Relu(Relu),

    /// [`LeakyRelu`] activation layer.
    LeakyRelu(LeakyRelu),

    /// [`SwiGlu`] activation layer.
    SwiGlu(SwiGlu),

    /// [`Selu`] activation layer.
    Selu(Selu),

    /// [`Sigmoid`] activation layer.
    Sigmoid(Sigmoid),

    /// [`Tanh`] activation layer.
    Tanh(Tanh),

    /// [`HardSigmoid`] activation layer.
    HardSigmoid(HardSigmoid),

    /// [`HardSwish`] activation layer.
    HardSwish(HardSwish),

    /// [`Softplus`] activation layer.
    Softplus(Softplus),

    /// [`Softsign`] activation layer.
    Softsign(Softsign),

    /// [`Elu`] activation layer.
    Elu(Elu),

    /// [`Celu`] activation layer.
    Celu(Celu),

    /// [`ThresholdedRelu`] activation layer.
    ThresholdedRelu(ThresholdedRelu),

    /// [`HardShrink`] activation layer.
    HardShrink(HardShrink),

    /// [`SoftShrink`] activation layer.
    SoftShrink(SoftShrink),

    /// [`Shrink`] activation layer.
    Shrink(Shrink),
}

impl From<Gelu> for Activation {
    fn from(layer: Gelu) -> Self {
        Self::Gelu(layer)
    }
}

impl From<PRelu> for Activation {
    fn from(layer: PRelu) -> Self {
        Self::PRelu(layer)
    }
}

impl From<Relu> for Activation {
    fn from(layer: Relu) -> Self {
        Self::Relu(layer)
    }
}

impl From<LeakyRelu> for Activation {
    fn from(layer: LeakyRelu) -> Self {
        Self::LeakyRelu(layer)
    }
}

impl From<SwiGlu> for Activation {
    fn from(layer: SwiGlu) -> Self {
        Self::SwiGlu(layer)
    }
}

impl From<Selu> for Activation {
    fn from(layer: Selu) -> Self {
        Self::Selu(layer)
    }
}

impl From<Sigmoid> for Activation {
    fn from(layer: Sigmoid) -> Self {
        Self::Sigmoid(layer)
    }
}

impl From<Tanh> for Activation {
    fn from(layer: Tanh) -> Self {
        Self::Tanh(layer)
    }
}

impl From<HardSigmoid> for Activation {
    fn from(layer: HardSigmoid) -> Self {
        Self::HardSigmoid(layer)
    }
}

impl From<HardSwish> for Activation {
    fn from(layer: HardSwish) -> Self {
        Self::HardSwish(layer)
    }
}

impl From<Softplus> for Activation {
    fn from(layer: Softplus) -> Self {
        Self::Softplus(layer)
    }
}

impl From<Softsign> for Activation {
    fn from(layer: Softsign) -> Self {
        Self::Softsign(layer)
    }
}

impl From<Elu> for Activation {
    fn from(layer: Elu) -> Self {
        Self::Elu(layer)
    }
}

impl From<Celu> for Activation {
    fn from(layer: Celu) -> Self {
        Self::Celu(layer)
    }
}

impl From<ThresholdedRelu> for Activation {
    fn from(layer: ThresholdedRelu) -> Self {
        Self::ThresholdedRelu(layer)
    }
}

impl From<HardShrink> for Activation {
    fn from(layer: HardShrink) -> Self {
        Self::HardShrink(layer)
    }
}

impl From<SoftShrink> for Activation {
    fn from(layer: SoftShrink) -> Self {
        Self::SoftShrink(layer)
    }
}

impl From<Shrink> for Activation {
    fn from(layer: Shrink) -> Self {
        Self::Shrink(layer)
    }
}

impl Activation {
    /// Forward pass.
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        match self {
            Activation::Relu(layer) => layer.forward(input),
            Activation::LeakyRelu(layer) => layer.forward(input),
            Activation::Gelu(layer) => layer.forward(input),
            Activation::PRelu(layer) => layer.forward(input),
            Activation::SwiGlu(layer) => layer.forward(input),
            Activation::HardSigmoid(layer) => layer.forward(input),
            Activation::HardSwish(layer) => layer.forward(input),
            Activation::Softplus(layer) => layer.forward(input),
            Activation::Selu(layer) => layer.forward(input),
            Activation::Sigmoid(layer) => layer.forward(input),
            Activation::Tanh(layer) => layer.forward(input),
            Activation::Softsign(layer) => layer.forward(input),
            Activation::Elu(layer) => layer.forward(input),
            Activation::Celu(layer) => layer.forward(input),
            Activation::ThresholdedRelu(layer) => layer.forward(input),
            Activation::HardShrink(layer) => layer.forward(input),
            Activation::SoftShrink(layer) => layer.forward(input),
            Activation::Shrink(layer) => layer.forward(input),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Module;

    fn make_input(device: &Device) -> Tensor<2> {
        Tensor::from_data([[-1.0, -0.5, 0.0], [1.0, 0.5, 0.0]], device)
    }

    fn expect_tensor<const D: usize>(actual: Tensor<D>, expected: Tensor<D>) {
        actual.to_data().assert_eq(&expected.to_data(), true);
    }

    fn check_stateless_config_output<const D: usize>(
        config: ActivationConfig,
        input: Tensor<D>,
        expected: Tensor<D>,
        device: &Device,
    ) {
        let act = config.init(device);
        let output = act.forward(input);
        expect_tensor(output, expected);
    }

    #[test]
    fn test_gelu() {
        let device = Default::default();
        let input = make_input(&device);

        let expected = Gelu::new().forward(input.clone());

        check_stateless_config_output(ActivationConfig::Gelu, input, expected, &device)
    }

    #[test]
    fn test_gelu_approximate() {
        let device = Default::default();
        let input = make_input(&device);

        let expected = Gelu::new_approximate().forward(input.clone());

        check_stateless_config_output(ActivationConfig::GeluApproximate, input, expected, &device)
    }

    #[test]
    fn test_prelu() {
        let device = Default::default();
        let input = make_input(&device);

        let inner_config = PReluConfig::new();
        let expected = inner_config.init(&device).forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_relu() {
        let device = Default::default();
        let input = make_input(&device);

        let expected = Relu.forward(input.clone());

        check_stateless_config_output(ActivationConfig::Relu, input, expected, &device)
    }

    #[test]
    fn test_leaky_relu() {
        let device = Default::default();
        let input = make_input(&device);

        let inner_config = LeakyReluConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_swi_glu() {
        let device = Default::default();
        let input = make_input(&device);

        let d_input = input.shape()[1];
        let d_output = 2 * d_input;

        let inner_config = SwiGluConfig::new(d_input, d_output);
        let mut reference = inner_config.init(&device);

        let config: ActivationConfig = inner_config.into();
        let layer = config.init(&device);

        // Access tensors via forward pass to trigger lazy initialization, then clone weights.
        let layer_output = layer.forward(input.clone());

        match &layer {
            Activation::SwiGlu(inner) => {
                let state = inner.clone().into_record();
                reference = reference.load_record(state);
            }
            _ => unreachable!(),
        };

        expect_tensor(layer_output, reference.forward(input.clone()))
    }

    #[test]
    fn test_selu() {
        let device = Default::default();
        let input = make_input(&device);

        let expected = Selu.forward(input.clone());

        check_stateless_config_output(ActivationConfig::Selu, input, expected, &device)
    }

    #[test]
    fn test_sigmoid() {
        let device = Default::default();
        let input = make_input(&device);

        let expected = Sigmoid.forward(input.clone());

        check_stateless_config_output(ActivationConfig::Sigmoid, input, expected, &device)
    }

    #[test]
    fn test_tanh() {
        let device = Default::default();
        let input = make_input(&device);

        let expected = Tanh.forward(input.clone());

        check_stateless_config_output(ActivationConfig::Tanh, input, expected, &device)
    }

    #[test]
    fn test_hard_sigmoid() {
        let device = Default::default();
        let input = make_input(&device);

        let inner_config = HardSigmoidConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_softsign() {
        let device = Default::default();
        let input = make_input(&device);

        let expected = Softsign.forward(input.clone());

        check_stateless_config_output(ActivationConfig::Softsign, input, expected, &device)
    }

    #[test]
    fn test_elu() {
        let device = Default::default();
        let input = make_input(&device);

        let inner_config = EluConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_softplus() {
        let device = Default::default();
        let input = make_input(&device);

        let inner_config = SoftplusConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_celu() {
        let device = Default::default();
        let input = make_input(&device);

        let inner_config = CeluConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_thresholded_relu() {
        let device = Default::default();
        let input = make_input(&device);

        let inner_config = ThresholdedReluConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_hard_shrink() {
        let device = Default::default();
        let input = make_input(&device);

        let inner_config = HardShrinkConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_soft_shrink() {
        let device = Default::default();
        let input = make_input(&device);

        let inner_config = SoftShrinkConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_shrink() {
        let device = Default::default();
        let input = make_input(&device);

        let inner_config = ShrinkConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }
}
