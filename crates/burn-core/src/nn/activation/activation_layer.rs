use crate as burn;
use burn_derive::{Config, Module};

use crate::nn::{
    Gelu, HardSigmoid, HardSigmoidConfig, LeakyRelu, LeakyReluConfig, Linear, LinearConfig, PRelu,
    PReluConfig, Relu, Sigmoid, SwiGlu, SwiGluConfig, Tanh,
};
use burn_tensor::Tensor;
use burn_tensor::backend::Backend;

/// [`ActivationLayer`] Configuration.
#[derive(Config, Debug)]
#[non_exhaustive]
pub enum ActivationLayerConfig {
    // TODO: GLU
    // GLU's dim-select interaction with DimSelectActivationLayer needs thought.
    /// [`Gelu`] activation layer.
    GeLu,

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

    /// [`Linear`] activation layer.
    Linear(LinearConfig),
}

impl ActivationLayerConfig {
    /// Initialize a wrapped activation layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ActivationLayer<B> {
        match self {
            ActivationLayerConfig::Relu => ActivationLayer::Relu(Relu),
            ActivationLayerConfig::LeakyRelu(conf) => ActivationLayer::LeakyRelu(conf.init()),
            ActivationLayerConfig::GeLu => ActivationLayer::Gelu(Gelu),
            ActivationLayerConfig::PRelu(conf) => ActivationLayer::PRelu(conf.init(device)),
            ActivationLayerConfig::SwiGlu(conf) => ActivationLayer::SwiGlu(conf.init(device)),
            ActivationLayerConfig::HardSigmoid(conf) => ActivationLayer::HardSigmoid(conf.init()),
            ActivationLayerConfig::Sigmoid => ActivationLayer::Sigmoid(Sigmoid),
            ActivationLayerConfig::Tanh => ActivationLayer::Tanh(Tanh),
            ActivationLayerConfig::Linear(conf) => ActivationLayer::Linear(conf.init(device)),
        }
    }
}

/// Activation Layer Wrapper.
///
/// Provides support for many in-built `burn::nn` activations.
#[derive(Module, Debug)]
#[non_exhaustive]
pub enum ActivationLayer<B: Backend> {
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

    /// [`Linear`] activation layer.
    Linear(Linear<B>),
}

impl<B: Backend> ActivationLayer<B> {
    /// Forward pass.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            ActivationLayer::Relu(layer) => layer.forward(input),
            ActivationLayer::LeakyRelu(layer) => layer.forward(input),
            ActivationLayer::Gelu(layer) => layer.forward(input),
            ActivationLayer::PRelu(layer) => layer.forward(input),
            ActivationLayer::SwiGlu(layer) => layer.forward(input),
            ActivationLayer::HardSigmoid(layer) => layer.forward(input),
            ActivationLayer::Sigmoid(layer) => layer.forward(input),
            ActivationLayer::Tanh(layer) => layer.forward(input),
            ActivationLayer::Linear(layer) => layer.forward(input),
        }
    }
}

/// [`DimSelectActivationLayer`] Config.
#[derive(Config, Debug)]
pub struct DimSelectActivationLayerConfig {
    /// Configuration of the inner layer.
    pub layer: ActivationLayerConfig,

    /// The activation dimension of the input.
    /// Supports negative indexing.
    #[config(default = "-1")]
    pub dim: isize,
}

impl DimSelectActivationLayerConfig {
    /// Initialize a [`DimSelectActivationLayer`].
    pub fn init<B: Backend>(&self, device: &B::Device) -> DimSelectActivationLayer<B> {
        DimSelectActivationLayer {
            layer: self.layer.init(device),
            dim: self.dim,
        }
    }
}

/// [`ActivationLayer`] wrapper with `dim`-select support.
///
/// Swaps the specified `dim` to the last dimension, applies the activation, then swaps back.
#[derive(Module, Debug)]
pub struct DimSelectActivationLayer<B: Backend> {
    /// Configuration of the inner layer.
    pub layer: ActivationLayer<B>,

    /// The activation dimension of the input.
    /// Supports negative indexing.
    pub dim: isize,
}

impl<B: Backend> DimSelectActivationLayer<B> {
    /// Canonicalize the activation dim to the given rank.
    #[inline(always)]
    pub fn canonicalize_dim(&self, rank: usize) -> usize {
        burn_tensor::indexing::canonicalize_dim(self.dim, rank, false)
    }

    /// Forward pass.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let dim = self.canonicalize_dim(D);
        let last = D - 1;

        let input = if dim == last {
            input
        } else {
            input.swap_dims(dim, last)
        };

        let output = self.layer.forward(input);

        if dim == last {
            output
        } else {
            output.swap_dims(dim, last)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use crate::prelude::Module;
    use burn_tensor::Distribution;
    use burn_tensor::activation::relu;

    fn make_input<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
        Tensor::from_data([[-1.0, -0.5, 0.0], [1.0, 0.5, 0.0]], device)
    }

    fn expect_tensor<B: Backend, const D: usize>(actual: Tensor<B, D>, expected: Tensor<B, D>) {
        actual.to_data().assert_eq(&expected.to_data(), true);
    }

    fn check_stateless_config_output<B: Backend, const D: usize>(
        config: ActivationLayerConfig,
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

        let expected = Gelu::default().forward(input.clone());

        check_stateless_config_output(ActivationLayerConfig::GeLu, input, expected, &device)
    }

    #[test]
    fn test_prelu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = PReluConfig::new();
        let expected = inner_config.init(&device).forward(input.clone());

        check_stateless_config_output(
            ActivationLayerConfig::PRelu(inner_config),
            input,
            expected,
            &device,
        )
    }

    #[test]
    fn test_relu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let expected = Relu::default().forward(input.clone());

        check_stateless_config_output(ActivationLayerConfig::Relu, input, expected, &device)
    }

    #[test]
    fn test_leaky_relu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = LeakyReluConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(
            ActivationLayerConfig::LeakyRelu(inner_config),
            input,
            expected,
            &device,
        )
    }

    #[test]
    fn test_swi_glu() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let d_input = input.shape().dims[1];
        let d_output = 2 * d_input;

        let inner_config = SwiGluConfig::new(d_input, d_output);
        let mut reference: SwiGlu<TestBackend> = inner_config.init(&device);

        let config = ActivationLayerConfig::SwiGlu(inner_config);
        let layer = config.init(&device);

        match &layer {
            ActivationLayer::SwiGlu(inner) => {
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

        let expected = Sigmoid::default().forward(input.clone());

        check_stateless_config_output(ActivationLayerConfig::Sigmoid, input, expected, &device)
    }

    #[test]
    fn test_tanh() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let expected = Tanh::default().forward(input.clone());

        check_stateless_config_output(ActivationLayerConfig::Tanh, input, expected, &device)
    }

    #[test]
    fn test_hard_sigmoid() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let inner_config = HardSigmoidConfig::new();
        let expected = inner_config.init().forward(input.clone());

        check_stateless_config_output(
            ActivationLayerConfig::HardSigmoid(inner_config),
            input,
            expected,
            &device,
        )
    }

    #[test]
    fn test_linear() {
        let device = Default::default();
        let input = make_input::<TestBackend>(&device);

        let d_input = input.shape().dims[1];
        let d_output = 2 * d_input;

        let inner_config = LinearConfig::new(d_input, d_output);
        let mut reference: Linear<TestBackend> = inner_config.init(&device);

        let config = ActivationLayerConfig::Linear(inner_config);
        let layer = config.init(&device);

        match &layer {
            ActivationLayer::Linear(inner) => {
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
    fn test_dim_select_activation_layer_default() {
        let device = Default::default();

        let input: Tensor<TestBackend, 3> =
            Tensor::random([2, 4, 3], Distribution::Normal(0.0, 1.0), &device);

        let expected = Relu::default().forward(input.clone());

        let config = DimSelectActivationLayerConfig::new(ActivationLayerConfig::Relu);
        let act = config.init(&device);
        let output = act.forward(input);
        expect_tensor(output, expected);
    }

    #[test]
    fn test_reproduce_swap_dims_bug() {
        // This is broken; see: https://github.com/tracel-ai/burn/issues/3602
        type B = TestBackend;
        let device = Default::default();

        // This appears to be shape-dependent; there are shapes for which the bug is not triggered.

        let input: Tensor<B, 3> =
            Tensor::random([2, 5, 4], Distribution::Normal(0.0, 1.0), &device);
        println!("input: {:?}\n", input);

        let swapped_input = input.clone().swap_dims(1, 2);
        println!("swapped_input: {:?}\n", swapped_input);

        let coherent: Tensor<B, 3> = Tensor::from_data(swapped_input.to_data(), &device);
        println!("coherent_input: {:?}\n", coherent);

        let swapped_result = relu(swapped_input.clone());
        println!("swapped result: {:?}\n", swapped_result);

        let coherent_result = relu(coherent);
        println!("coherent result: {:?}\n", coherent_result);

        swapped_result
            .to_data()
            .assert_eq(&coherent_result.to_data(), true);
    }

    #[test]
    fn test_dim_select_activation_layer_with_dim() {
        // This is broken; see: https://github.com/tracel-ai/burn/issues/3602
        let device = Default::default();

        let input: Tensor<TestBackend, 3> =
            Tensor::random([2, 5, 4], Distribution::Normal(0.0, 1.0), &device);
        println!("input: {:?}\n", input);

        let swapped_input = input.clone().swap_dims(1, 2);
        println!("input.swap: {:?}\n", swapped_input);
        let native_result = Relu::default().forward(swapped_input);
        println!("native output: {:?}\n", native_result);
        let expected = native_result.swap_dims(1, 2);
        println!("expected: {:?}\n", expected);

        let config = DimSelectActivationLayerConfig::new(ActivationLayerConfig::Relu).with_dim(1);
        let act = config.init(&device);
        assert_eq!(act.canonicalize_dim(3), 1);

        let result = act.forward(input);
        println!("result: {:?}\n", result);
        expect_tensor(result, expected);
    }
}
