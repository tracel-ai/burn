use crate as burn;
use burn_derive::{Config, Module};

use crate::nn::activation::{
    Gelu, HardSigmoid, HardSigmoidConfig, LeakyRelu, LeakyReluConfig, PRelu, PReluConfig, Relu,
    Sigmoid, SwiGlu, SwiGluConfig, Tanh,
};
use burn_tensor::backend::Backend;
use burn_tensor::{AsIndex, Tensor};

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

impl ActivationConfig {
    /// Initialize a wrapped activation layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Activation<B> {
        match self {
            ActivationConfig::Relu => Activation::Relu(Relu),
            ActivationConfig::LeakyRelu(conf) => Activation::LeakyRelu(conf.init()),
            ActivationConfig::Gelu => Activation::Gelu(Gelu),
            ActivationConfig::PRelu(conf) => Activation::PRelu(conf.init(device)),
            ActivationConfig::SwiGlu(conf) => Activation::SwiGlu(conf.init(device)),
            ActivationConfig::HardSigmoid(conf) => Activation::HardSigmoid(conf.init()),
            ActivationConfig::Sigmoid => Activation::Sigmoid(Sigmoid),
            ActivationConfig::Tanh => Activation::Tanh(Tanh),
        }
    }
}

/// Activation Layer Wrapper.
///
/// Provides support for many in-built `burn::nn` activations.
#[derive(Module, Debug)]
#[non_exhaustive]
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
            Activation::Sigmoid(layer) => layer.forward(input),
            Activation::Tanh(layer) => layer.forward(input),
        }
    }
}

/// [`DimSelectActivation`] Config.
#[derive(Config, Debug)]
pub struct DimSelectActivationConfig {
    /// Configuration of the inner layer.
    pub layer: ActivationConfig,

    /// The activation dimension of the input.
    /// Supports negative indexing.
    #[config(default = "-1")]
    pub dim: isize,
}

impl From<ActivationConfig> for DimSelectActivationConfig {
    fn from(config: ActivationConfig) -> Self {
        Self::new(config)
    }
}

impl DimSelectActivationConfig {
    /// Initialize a [`DimSelectActivation`].
    pub fn init<B: Backend>(&self, device: &B::Device) -> DimSelectActivation<B> {
        DimSelectActivation {
            layer: self.layer.init(device),
            dim: self.dim,
        }
    }
}

/// [`Activation`] wrapper with `dim`-select support.
///
/// Swaps the specified `dim` to the last dimension, applies the activation, then swaps back.
#[derive(Module, Debug)]
pub struct DimSelectActivation<B: Backend> {
    /// Configuration of the inner layer.
    pub layer: Activation<B>,

    /// The activation dimension of the input.
    /// Supports negative indexing.
    pub dim: isize,
}

impl<B: Backend> DimSelectActivation<B> {
    /// Forward pass.
    ///
    /// Swaps the input dims for the selected activation dim,
    /// applies the inner activation wrapper,
    /// then swaps the result dims back.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        apply_swapped(input, self.dim, |input| self.layer.forward(input))
    }
}

/// Swap the specified `dim` to the last dimension, apply the activation, then swap back.
///
/// # Arguments
///
/// - `input`: the tensor to operate on.
/// - `dim`: the dim to make the "last" dim.
/// - `f`: the function to apply.
///
/// # Returns
///
/// The result tensor.
fn apply_swapped<B: Backend, const D: usize, Dim, F>(
    input: Tensor<B, D>,
    dim: Dim,
    f: F,
) -> Tensor<B, D>
where
    Dim: AsIndex,
    F: FnOnce(Tensor<B, D>) -> Tensor<B, D>,
{
    let dim = burn_tensor::indexing::canonicalize_dim(dim, D, false);
    let last = D - 1;
    // swap_dims(x, x) is a guaranteed no-op.
    f(input.swap_dims(dim, last)).swap_dims(dim, last)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use crate::prelude::Module;
    use burn_tensor::{Distribution, TensorData};

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
    fn test_apply_swapped() {
        let device = Default::default();
        let input: Tensor<TestBackend, 2> =
            Tensor::from_data([[-1.0, -0.5, 0.0], [1.0, 0.5, 0.0]], &device);

        let output = apply_swapped(input.clone(), 1, |t| {
            t + Tensor::from_data([[2.0, 4.0, 6.0]], &device)
        });
        output.clone().to_data().assert_eq(
            &TensorData::from([
                [2.0 - 1.0, 4.0 - 0.5, 6.0 - 0.0],
                [2.0 + 1.0, 4.0 + 0.5, 6.0 + 0.0],
            ]),
            false,
        );
        // Test negative dim.
        output.clone().to_data().assert_eq(
            &apply_swapped(input.clone(), -1, |t| {
                t + Tensor::from_data([[2.0, 4.0, 6.0]], &device)
            })
            .to_data(),
            true,
        );

        let output = apply_swapped(input.clone(), 0, |t| {
            t + Tensor::from_data([[2.0, 4.0]], &device)
        });
        output.to_data().assert_eq(
            &TensorData::from([
                [2.0 - 1.0, 2.0 - 0.5, 2.0 - 0.0],
                [4.0 + 1.0, 4.0 + 0.5, 4.0 + 0.0],
            ]),
            false,
        );
    }

    #[test]
    fn test_dim_select_activation_layer_default() {
        let device = Default::default();

        let input: Tensor<TestBackend, 3> =
            Tensor::random([2, 4, 3], Distribution::Normal(0.0, 1.0), &device);

        let expected = Relu.forward(input.clone());

        let config: DimSelectActivationConfig = ActivationConfig::Relu.into();
        let act = config.init(&device);
        let output = act.forward(input);
        expect_tensor(output, expected);
    }

    #[test]
    fn test_dim_select_activation_layer_with_dim() {
        // This is broken; see: https://github.com/tracel-ai/burn/issues/3602
        let device = Default::default();

        let input: Tensor<TestBackend, 3> =
            Tensor::random([2, 5, 4], Distribution::Normal(0.0, 1.0), &device);

        let expected = Relu.forward(input.clone().swap_dims(1, 2)).swap_dims(1, 2);

        let config = DimSelectActivationConfig::from(ActivationConfig::Relu).with_dim(1);
        let act = config.init(&device);

        let result = act.forward(input);
        expect_tensor(result, expected);
    }
}
