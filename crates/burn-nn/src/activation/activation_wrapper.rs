use burn_core as burn;

use crate::activation::{
    Celu, CeluConfig, Elu, EluConfig, Gelu, HardShrink, HardShrinkConfig, HardSigmoid,
    HardSigmoidConfig, HardSwish, LeakyRelu, LeakyReluConfig, PRelu, PReluConfig, Relu, Selu,
    Shrink, ShrinkConfig, Sigmoid, SoftShrink, SoftShrinkConfig, Softplus, SoftplusConfig,
    Softsign, SwiGlu, SwiGluConfig, Tanh, ThresholdedRelu, ThresholdedReluConfig,
};
use burn::config::Config;
use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

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
    pub fn init<B: Backend>(&self, device: &B::Device) -> Activation<B> {
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
pub enum Activation<B: Backend> {
    /// [`Gelu`] activation layer.
    Gelu(Gelu),

    /// [`PRelu`] activation layer.
    PRelu(PRelu<B>),

    /// [`Relu`] activation layer.
    Relu(Relu),

    /// [`LeakyRelu`] activation layer.
    LeakyRelu(LeakyRelu),

    /// [`SwiGlu`] activation layer.
    SwiGlu(SwiGlu<B>),

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

impl<B: Backend> From<Gelu> for Activation<B> {
    fn from(layer: Gelu) -> Self {
        Self::Gelu(layer)
    }
}

impl<B: Backend> From<PRelu<B>> for Activation<B> {
    fn from(layer: PRelu<B>) -> Self {
        Self::PRelu(layer)
    }
}

impl<B: Backend> From<Relu> for Activation<B> {
    fn from(layer: Relu) -> Self {
        Self::Relu(layer)
    }
}

impl<B: Backend> From<LeakyRelu> for Activation<B> {
    fn from(layer: LeakyRelu) -> Self {
        Self::LeakyRelu(layer)
    }
}

impl<B: Backend> From<SwiGlu<B>> for Activation<B> {
    fn from(layer: SwiGlu<B>) -> Self {
        Self::SwiGlu(layer)
    }
}

impl<B: Backend> From<Selu> for Activation<B> {
    fn from(layer: Selu) -> Self {
        Self::Selu(layer)
    }
}

impl<B: Backend> From<Sigmoid> for Activation<B> {
    fn from(layer: Sigmoid) -> Self {
        Self::Sigmoid(layer)
    }
}

impl<B: Backend> From<Tanh> for Activation<B> {
    fn from(layer: Tanh) -> Self {
        Self::Tanh(layer)
    }
}

impl<B: Backend> From<HardSigmoid> for Activation<B> {
    fn from(layer: HardSigmoid) -> Self {
        Self::HardSigmoid(layer)
    }
}

impl<B: Backend> From<HardSwish> for Activation<B> {
    fn from(layer: HardSwish) -> Self {
        Self::HardSwish(layer)
    }
}

impl<B: Backend> From<Softplus> for Activation<B> {
    fn from(layer: Softplus) -> Self {
        Self::Softplus(layer)
    }
}

impl<B: Backend> From<Softsign> for Activation<B> {
    fn from(layer: Softsign) -> Self {
        Self::Softsign(layer)
    }
}

impl<B: Backend> From<Elu> for Activation<B> {
    fn from(layer: Elu) -> Self {
        Self::Elu(layer)
    }
}

impl<B: Backend> From<Celu> for Activation<B> {
    fn from(layer: Celu) -> Self {
        Self::Celu(layer)
    }
}

impl<B: Backend> From<ThresholdedRelu> for Activation<B> {
    fn from(layer: ThresholdedRelu) -> Self {
        Self::ThresholdedRelu(layer)
    }
}

impl<B: Backend> From<HardShrink> for Activation<B> {
    fn from(layer: HardShrink) -> Self {
        Self::HardShrink(layer)
    }
}

impl<B: Backend> From<SoftShrink> for Activation<B> {
    fn from(layer: SoftShrink) -> Self {
        Self::SoftShrink(layer)
    }
}

impl<B: Backend> From<Shrink> for Activation<B> {
    fn from(layer: Shrink) -> Self {
        Self::Shrink(layer)
    }
}

impl<B: Backend> Activation<B> {
    /// Forward pass.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
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
    use crate::TestBackend;
    use burn::module::Module;

    fn make_input<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
        Tensor::from_data([[-1.0, -0.5, 0.0], [1.0, 0.5, 0.0]], device)
    }

    fn expect_tensor<B: Backend, const D: usize>(actual: Tensor<B, D>, expected: Tensor<B, D>) {
        actual.to_data().assert_eq(&expected.to_data(), true);
    }

    fn check_stateless_config_output<B: Backend, const D: usize>(
        config: ActivationConfig,
        input: Tensor<B, D>,
        expected: Tensor<B, D>,
        device: &B::Device,
    ) {
        let act = config.init(device);
        let output = act.forward(input);
        expect_tensor(output, expected);
    }

    #[test]
    fn test_gelu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let expected = Gelu::new().forward(input.clone());

        check_stateless_config_output(ActivationConfig::Gelu, input, expected, &device)
    }

    #[test]
    fn test_gelu_approximate() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let expected = Gelu::new_approximate().forward(input.clone());

        check_stateless_config_output(ActivationConfig::GeluApproximate, input, expected, &device)
    }

    #[test]
    fn test_prelu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = PReluConfig::new();
        let expected = inner_config.init(&device).forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_relu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let expected = Relu.forward(input.clone());

        check_stateless_config_output(ActivationConfig::Relu, input, expected, &device)
    }

    #[test]
    fn test_leaky_relu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = LeakyReluConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_swi_glu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let d_input = input.shape()[1];
        let d_output = 2 * d_input;

        let inner_config = SwiGluConfig::new(d_input, d_output);
        let mut reference: SwiGlu<TestBackend> = inner_config.init(&device);

        let config: ActivationConfig = inner_config.into();
        let layer = config.init(&device);

        match &layer {
            Activation::SwiGlu(inner) => {
                // Clone the initialized weights.
                let state = inner.clone().into_record();
                reference = reference.load_record(state);
            }
            _ => unreachable!(),
        };

        expect_tensor(
            layer.forward(input.clone()),
            reference.forward(input.clone()),
        )
    }

    #[test]
    fn test_selu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let expected = Selu.forward(input.clone());

        check_stateless_config_output(ActivationConfig::Selu, input, expected, &device)
    }

    #[test]
    fn test_sigmoid() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let expected = Sigmoid.forward(input.clone());

        check_stateless_config_output(ActivationConfig::Sigmoid, input, expected, &device)
    }

    #[test]
    fn test_tanh() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let expected = Tanh.forward(input.clone());

        check_stateless_config_output(ActivationConfig::Tanh, input, expected, &device)
    }

    #[test]
    fn test_hard_sigmoid() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = HardSigmoidConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_softsign() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let expected = Softsign.forward(input.clone());

        check_stateless_config_output(ActivationConfig::Softsign, input, expected, &device)
    }

    #[test]
    fn test_elu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = EluConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_softplus() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = SoftplusConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_celu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = CeluConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_thresholded_relu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = ThresholdedReluConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_hard_shrink() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = HardShrinkConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_soft_shrink() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = SoftShrinkConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }

    #[test]
    fn test_shrink() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = ShrinkConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(inner_config.into(), input, expected, &device)
    }
}
