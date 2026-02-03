use burn_core as burn;

use crate::activation::{
    Celu, CeluConfig, Gelu, HardSigmoid, HardSigmoidConfig, HardSwish, LeakyRelu, LeakyReluConfig,
    PRelu, PReluConfig, Relu, Sigmoid, Softplus, SoftplusConfig, SwiGlu, SwiGluConfig, Tanh,
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

    /// [`PRelu`] activation layer.
    PRelu(PReluConfig),

    /// [`Relu`] activation layer.
    Relu,

    /// [`LeakyRelu`] activation layer.
    LeakyRelu(LeakyReluConfig),

    /// [`SwiGlu`] activation layer.
    SwiGlu(SwiGluConfig),

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

    /// [`Celu`] activation layer.
    Celu(CeluConfig),
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

impl From<CeluConfig> for ActivationConfig {
    fn from(config: CeluConfig) -> Self {
        Self::Celu(config)
    }
}

impl ActivationConfig {
    /// Initialize a wrapped activation layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Activation<B> {
        match self {
            ActivationConfig::Relu => Relu.into(),
            ActivationConfig::LeakyRelu(conf) => conf.init().into(),
            ActivationConfig::Gelu => Gelu.into(),
            ActivationConfig::PRelu(conf) => conf.init(device).into(),
            ActivationConfig::SwiGlu(conf) => conf.init(device).into(),
            ActivationConfig::HardSigmoid(conf) => conf.init().into(),
            ActivationConfig::HardSwish => HardSwish.into(),
            ActivationConfig::Softplus(conf) => conf.init().into(),
            ActivationConfig::Sigmoid => Sigmoid.into(),
            ActivationConfig::Tanh => Tanh.into(),
            ActivationConfig::Celu(conf) => conf.init().into(),
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

    /// [`Celu`] activation layer.
    Celu(Celu),
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

impl<B: Backend> From<Celu> for Activation<B> {
    fn from(layer: Celu) -> Self {
        Self::Celu(layer)
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
            Activation::Sigmoid(layer) => layer.forward(input),
            Activation::Tanh(layer) => layer.forward(input),
            Activation::Celu(layer) => layer.forward(input),
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

        let expected = Gelu.forward(input.clone());

        check_stateless_config_output(ActivationConfig::Gelu, input, expected, &device)
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

        let d_input = input.shape().dims[1];
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
}
