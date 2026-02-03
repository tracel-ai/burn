use burn_core as burn;

use crate::GateController;
use crate::activation::{Activation, ActivationConfig};
use burn::config::Config;
use burn::module::{Content, DisplaySettings, Initializer, Module, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// A RnnState is used to store hidden state in RNN.
pub struct RnnState<B: Backend, const D: usize> {
    /// The hidden state. 
    pub hidden: Tensor<B, D>,
}

impl<B: Backend, const D: usize> RnnState<B, D> {
    /// Initialize a new [RNN State](RnnState).
    pub fn new(hidden: Tensor<B, D>) -> Self {
        Self { hidden }
    }
}

/// Configuration to create a [Rnn](Rnn) module using the [init function](RnnConfig::init).
#[derive(Config, Debug)]
pub struct RnnConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If a bias should be applied during the Rnn transformation.
    pub bias: bool,
    /// Rnn initializer
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    pub initializer: Initializer,
    /// If true, the input tensor is expected to be `[batch_size, seq_length, input_size]`.
    /// If false, the input tensor is expected to be `[seq_length, batch_size, input_size]`.
    #[config(default = true)]
    pub batch_first: bool,
    /// If true, process the sequence in reverse order.
    /// This is useful for implementing reverse-direction RNNs (e.g., ONNX reverse direction).
    #[config(default = false)]
    pub reverse: bool,
    /// Optional cell state clip threshold. If provided, cell state values are clipped
    /// to the range `[-clip, +clip]` after each timestep. This can help prevent
    /// exploding values during inference.
    pub clip: Option<f64>,
    /// Activation function applied to the cell state before computing hidden output.
    /// Default is Tanh, which is standard for Rnn.
    #[config(default = "ActivationConfig::Tanh")]
    pub hidden_activation: ActivationConfig,
}


/// The Rnn module. This implementation is for a unidirectional, stateless, Rnn.
/// Should be created with [RnnConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct Rnn<B: Backend> {
    /// gate controller for Rnn (has single gate).
    pub gate: GateController<B>,
    /// The hidden state of the Rnn.
    pub d_hidden: usize,
    /// If true, input is `[batch_size, seq_length, input_size]`.
    /// If false, input is `[seq_length, batch_size, input_size]`.
    pub batch_first: bool,
    /// If true, process the sequence in reverse order.
    pub reverse: bool,
    /// Optional cell state clip threshold.
    pub clip: Option<f64>,
    /// Activation function for hidden output.
    pub hidden_activation: Activation<B>,
}

impl<B: Backend> ModuleDisplay for Rnn<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }
    
    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_input, _] = self.gate.input_transform.weight.shape().dims();
        let bias = self.gate.input_transform.bias.is_some();

        content
            .add("d_input", &d_input)
            .add("d_hidden", &self.d_hidden)
            .add("bias", &bias)
            .optional()
    }
}

impl RnnConfig {
    /// Initialize a new [Rnn](Rnn) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Rnn<B> {
        let d_output = self.d_hidden;

        let new_gate = || {
            GateController::new(
                self.d_input,
                d_output,
                self.bias,
                self.initializer.clone(),
                device,
            )
        };

        Rnn {
            gate: new_gate(),
            d_hidden: self.d_hidden,
            batch_first: self.batch_first,
            reverse: self.reverse,
            clip: self.clip,
            hidden_activation: self.hidden_activation.init(device),
        }
    }
}

impl<B: Backend> Rnn<B> {
    /// Applies the forward pass on the input tensor. This RNN implementation
    /// returns the state for each element in a sequence (i.e., across seq_length) and a final state.
    ///
    /// ## Parameters:
    /// - batched_input: The input tensor of shape:
    ///   - `[batch_size, sequence_length, input_size]` if `batch_first` is true (default)
    ///   - `[sequence_length, batch_size, input_size]` if `batch_first` is false
    /// - state: An optional `RnnState` representing the initial hidden state.
    ///   The state tensor has shape `[batch_size, hidden_size]`.
    ///   If no initial state is provided, these tensors are initialized to zeros.
    ///
    /// ## Returns:
    /// - output: A tensor represents the output features of Rnn. Shape:
    ///   - `[batch_size, sequence_length, hidden_size]` if `batch_first` is true
    ///   - `[sequence_length, batch_size, hidden_size]` if `batch_first` is false
    /// - state: A `RnnState` represents the final hidden state. The hidden state tensor has the shape
    ///   `[batch_size, hidden_size]`.
    pub fn forward(
        &self,
        batched_input: Tensor<B, 3>,
        state: Option<RnnState<B, 2>>,
    ) -> (Tensor<B, 3>, RnnState<B, 2>) {
        // Convert to batch-first layout internally if needed
        let batched_input = if self.batch_first {
            batched_input
        } else {
            batched_input.swap_dims(0, 1)
        };

        let device = batched_input.device();
        let [batch_size, seq_length, _] = batched_input.dims();

        // Process sequence in forward or reverse order based on config
        let (output, state) = if self.reverse {
            self.forward_iter(
                batched_input.iter_dim(1).rev().zip((0..seq_length).rev()),
                state,
                batch_size,
                seq_length,
                &device,
            )
        } else {
            self.forward_iter(
                batched_input.iter_dim(1).zip(0..seq_length),
                state,
                batch_size,
                seq_length,
                &device,
            )
        };

        // Convert output back to seq-first layout if needed
        let output = if self.batch_first {
            output
        } else {
            output.swap_dims(0, 1)
        };

        (output, state)
    }

    fn forward_iter<I: Iterator<Item = (Tensor<B, 3>, usize)>>(
        &self,
        input_timestep_iter: I,
        state: Option<RnnState<B, 2>>,
        batch_size: usize,
        seq_length: usize,
        device: &B::Device,
    ) -> (Tensor<B, 3>, RnnState<B, 2>) {
        let mut batched_hidden_state =
            Tensor::empty([batch_size, seq_length, self.d_hidden], device);

        let mut hidden_state = match state {
            Some(state) => state.hidden,
            None => Tensor::zeros([batch_size, self.d_hidden], device),
        };

        for (input_t, t) in input_timestep_iter {
            let input_t = input_t.squeeze_dim(1);

            // Compute gate output: h_t = activation(W_i @ x_t + W_h @ h_{t-1} + b)
            let biased_gate_sum = self
                .gate
                .gate_product(input_t.clone(), hidden_state.clone());
            
            let output_values = self.hidden_activation.forward(biased_gate_sum);
            
            // Update hidden state
            hidden_state = output_values;

            // Apply cell state clipping if configured
            if let Some(clip) = self.clip {
                hidden_state = hidden_state.clamp(-clip, clip);
            }

            let unsqueezed_hidden_state = hidden_state.clone().unsqueeze_dim(1);

            // store the hidden state for this timestep
            batched_hidden_state = batched_hidden_state.slice_assign(
                [0..batch_size, t..(t + 1), 0..self.d_hidden],
                unsqueezed_hidden_state.clone(),
            );
        }

        (
            batched_hidden_state,
            RnnState::new(hidden_state),
        )
    }
}

/// Configuration to create a [BiRnn](BiRnn) module using the [init function](BiRnnConfig::init).
#[derive(Config, Debug)]
pub struct BiRnnConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If a bias should be applied during the BiRnn transformation.
    pub bias: bool,
    /// BiRnn initializer
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    pub initializer: Initializer,
    /// If true, the input tensor is expected to be `[batch_size, seq_length, input_size]`.
    /// If false, the input tensor is expected to be `[seq_length, batch_size, input_size]`.
    #[config(default = true)]
    pub batch_first: bool,
    /// Optional cell state clip threshold.
    pub clip: Option<f64>,
    /// Activation function applied to the cell state before computing hidden output.
    #[config(default = "ActivationConfig::Tanh")]
    pub hidden_activation: ActivationConfig,
}

/// The BiRnn module. This implementation is for Bidirectional RNN.
/// Should be created with [BiRnnConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct BiRnn<B: Backend> {
    /// RNN for the forward direction.
    pub forward: Rnn<B>,
    /// RNN for the reverse direction.
    pub reverse: Rnn<B>,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If true, input is `[batch_size, seq_length, input_size]`.
    /// If false, input is `[seq_length, batch_size, input_size]`.
    pub batch_first: bool,
}

impl<B: Backend> ModuleDisplay for BiRnn<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_input, _] = self
            .forward
            .gate
            .input_transform
            .weight
            .shape()
            .dims();
        let bias = self.forward.gate.input_transform.bias.is_some();

        content
            .add("d_input", &d_input)
            .add("d_hidden", &self.d_hidden)
            .add("bias", &bias)
            .optional()
    }
}


impl BiRnnConfig {
    /// Initialize a new [Bidirectional RNN](BiRnn) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BiRnn<B> {
        // Internal RNNs always use batch_first=true; BiRnn handles layout conversion
        let base_config = RnnConfig::new(self.d_input, self.d_hidden, self.bias)
            .with_initializer(self.initializer.clone())
            .with_batch_first(true)
            .with_clip(self.clip)
            .with_hidden_activation(self.hidden_activation.clone());

        BiRnn {
            forward: base_config.clone().init(device),
            reverse: base_config.init(device),
            d_hidden: self.d_hidden,
            batch_first: self.batch_first,
        }
    }
}

impl<B: Backend> BiRnn<B> {
    /// Applies the forward pass on the input tensor. This Bidirectional RNN implementation
    /// returns the state for each element in a sequence (i.e., across seq_length) and a final state.
    ///
    /// ## Parameters:
    /// - batched_input: The input tensor of shape:
    ///   - `[batch_size, sequence_length, input_size]` if `batch_first` is true (default)
    ///   - `[sequence_length, batch_size, input_size]` if `batch_first` is false
    /// - state: An optional `RnnState` representing the hidden state.
    ///   Each state tensor has shape `[2, batch_size, hidden_size]`.
    ///   If no initial state is provided, these tensors are initialized to zeros.
    ///
    /// ## Returns:
    /// - output: A tensor represents the output features of RNN. Shape:
    ///   - `[batch_size, sequence_length, hidden_size * 2]` if `batch_first` is true
    ///   - `[sequence_length, batch_size, hidden_size * 2]` if `batch_first` is false
    /// - state: A `RnnState` represents the final forward and reverse states.
    ///   The `state.hidden` have the shape `[2, batch_size, hidden_size]`.
    pub fn forward(
        &self,
        batched_input: Tensor<B, 3>,
        state: Option<RnnState<B, 3>>,
    ) -> (Tensor<B, 3>, RnnState<B, 3>) {
        // Convert to batch-first layout internally if needed
        let batched_input = if self.batch_first {
            batched_input
        } else {
            batched_input.swap_dims(0, 1)
        };

        let device = batched_input.clone().device();
        let [batch_size, seq_length, _] = batched_input.shape().dims();

        let [init_state_forward, init_state_reverse] = match state {
            Some(state) => {
                let hidden_state_forward = state
                    .hidden
                    .clone()
                    .slice([0..1, 0..batch_size, 0..self.d_hidden])
                    .squeeze_dim(0);
                let hidden_state_reverse = state
                    .hidden
                    .slice([1..2, 0..batch_size, 0..self.d_hidden])
                    .squeeze_dim(0);

                [
                    Some(RnnState::new(hidden_state_forward)),
                    Some(RnnState::new(hidden_state_reverse)),
                ]
            }
            None => [None, None],
        };

        // forward direction
        let (batched_hidden_state_forward, final_state_forward) = self
            .forward
            .forward(batched_input.clone(), init_state_forward);

        // reverse direction
        let (batched_hidden_state_reverse, final_state_reverse) = self.reverse.forward_iter(
            batched_input.iter_dim(1).rev().zip((0..seq_length).rev()),
            init_state_reverse,
            batch_size,
            seq_length,
            &device,
        );

        let output = Tensor::cat(
            [batched_hidden_state_forward, batched_hidden_state_reverse].to_vec(),
            2,
        );

        // Convert output back to seq-first layout if needed
        let output = if self.batch_first {
            output
        } else {
            output.swap_dims(0, 1)
        };

        let state = RnnState::new(
            Tensor::stack(
                [final_state_forward.hidden, final_state_reverse.hidden].to_vec(),
                0,
            ),
        );

        (output, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LinearRecord, TestBackend};
    use burn::module::Param;
    use burn::tensor::{Device, Distribution, TensorData};
    use burn::tensor::{ElementConversion, Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[cfg(feature = "std")]
    use crate::TestAutodiffBackend;

    #[test]
    fn test_with_uniform_initializer() {
        let device = Default::default();
        TestBackend::seed(&device, 0);

        let config = RnnConfig::new(5, 5, false)
            .with_initializer(Initializer::Uniform { min: 0.0, max: 1.0 });
        let rnn = config.init::<TestBackend>(&Default::default());

        let gate_to_data =
            |gate: GateController<TestBackend>| gate.input_transform.weight.val().to_data();

        gate_to_data(rnn.gate).assert_within_range::<FT>(0.elem()..1.elem());
    }

    /// Test forward pass with simple input vector.
    ///
    /// Simple RNN: h_t = tanh(W_input @ x_t + W_hidden @ h_{t-1} + b)
    /// With input=0.1, weight_input=0.5, bias=0.0, h_0=0.2, weight_hidden=0.3
    /// h_t = tanh(0.5 * 0.1 + 0.3 * 0.2 + 0.0) = tanh(0.11) ≈ 0.10955
    #[test]
    fn test_forward_single_input_single_feature() {
        let device = Default::default();
        TestBackend::seed(&device, 0);

        let config = RnnConfig::new(1, 1, false);
        let device = Default::default();
        let mut rnn = config.init::<TestBackend>(&device);

        fn create_gate_controller(
            weights: f32,
            biases: f32,
            d_input: usize,
            d_output: usize,
            bias: bool,
            initializer: Initializer,
            device: &Device<TestBackend>,
        ) -> GateController<TestBackend> {
            let record_1 = LinearRecord {
                weight: Param::from_data(TensorData::from([[weights]]), device),
                bias: Some(Param::from_data(TensorData::from([biases]), device)),
            };
            let record_2 = LinearRecord {
                weight: Param::from_data(TensorData::from([[weights]]), device),
                bias: Some(Param::from_data(TensorData::from([biases]), device)),
            };
            GateController::create_with_weights(
                d_input,
                d_output,
                bias,
                initializer,
                record_1,
                record_2,
            )
        }

        rnn.gate = create_gate_controller(
            0.5,
            0.0,
            1,
            1,
            false,
            Initializer::XavierUniform { gain: 1.0 },
            &device,
        );

        // single timestep with single feature
        let input = Tensor::<TestBackend, 3>::from_data(TensorData::from([[[0.1]]]), &device);

        let (output, state) = rnn.forward(input, None);

        let tolerance = Tolerance::default();
        let expected = TensorData::from([[0.10955]]);
        state
            .hidden
            .to_data()
            .assert_approx_eq::<FT>(&expected, tolerance);

        output
            .select(0, Tensor::arange(0..1, &device))
            .squeeze_dim::<2>(0)
            .to_data()
            .assert_approx_eq::<FT>(&state.hidden.to_data(), tolerance);
    }

    #[test]
    fn test_batched_forward_pass_batch_of_one() {
        let device = Default::default();
        let rnn = RnnConfig::new(64, 1024, true).init(&device);
        let batched_input =
            Tensor::<TestBackend, 3>::random([1, 2, 64], Distribution::Default, &device);

        let (output, state) = rnn.forward(batched_input, None);
        assert_eq!(output.dims(), [1, 2, 1024]);
        assert_eq!(state.hidden.dims(), [1, 1024]);
    }


    #[test]
    #[cfg(feature = "std")]
    fn test_batched_backward_pass() {
        use burn::tensor::Shape;
        let device = Default::default();
        let rnn = RnnConfig::new(64, 32, true).init(&device);
        let shape: Shape = [8, 10, 64].into();
        let batched_input =
            Tensor::<TestAutodiffBackend, 3>::random(shape, Distribution::Default, &device);

        let (output, _) = rnn.forward(batched_input.clone(), None);
        let fake_loss = output;
        let grads = fake_loss.backward();

        let some_gradient = rnn
            .gate
            .hidden_transform
            .weight
            .grad(&grads)
            .unwrap();

        // Asserts that the gradients exist and are non-zero
        assert_ne!(
            some_gradient
                .any()
                .into_data()
                .iter::<f32>()
                .next()
                .unwrap(),
            0.0
        );
    }

        #[test]
    fn test_bidirectional() {
        let device = Default::default();
        TestBackend::seed(&device, 0);

        let config = BiRnnConfig::new(2, 3, true);
        let device = Default::default();
        let mut rnn = config.init(&device);

        fn create_gate_controller<const D1: usize, const D2: usize>(
            input_weights: [[f32; D1]; D2],
            input_biases: [f32; D1],
            hidden_weights: [[f32; D1]; D1],
            hidden_biases: [f32; D1],
            device: &Device<TestBackend>,
        ) -> GateController<TestBackend> {
            let d_input = input_weights[0].len();
            let d_output = input_weights.len();

            let input_record = LinearRecord {
                weight: Param::from_data(TensorData::from(input_weights), device),
                bias: Some(Param::from_data(TensorData::from(input_biases), device)),
            };
            let hidden_record = LinearRecord {
                weight: Param::from_data(TensorData::from(hidden_weights), device),
                bias: Some(Param::from_data(TensorData::from(hidden_biases), device)),
            };
            GateController::create_with_weights(
                d_input,
                d_output,
                true,
                Initializer::XavierUniform { gain: 1.0 },
                input_record,
                hidden_record,
            )
        }

        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[
                [0.949, -0.861],
                [0.892, 0.927],
                [-0.173, -0.301],
                [-0.081, 0.992],
            ]]),
            &device,
        );
        let h0 = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[[0.280, 0.360, -1.242]], [[-0.588, 0.729, -0.788]]]),
            &device,
        );

        rnn.forward.gate = create_gate_controller(
            [[0.367, 0.091, 0.342], [0.322, 0.533, 0.059]],
            [-0.196, 0.354, 0.209],
            [
                [-0.320, 0.232, -0.165],
                [0.093, -0.572, -0.315],
                [-0.467, 0.325, 0.046],
            ],
            [0.181, -0.190, -0.245],
            &device,
        );

        rnn.reverse.gate = create_gate_controller(
            [[-0.055, 0.506, 0.247], [-0.369, 0.178, -0.258]],
            [0.540, -0.164, 0.033],
            [
                [0.159, 0.180, -0.037],
                [-0.443, 0.485, -0.488],
                [0.098, -0.085, -0.140],
            ],
            [-0.510, 0.105, 0.114],
            &device,
        );

        let expected_output_with_init_state = TensorData::from([[
            [0.23764, -0.03442, 0.04414, -0.15635, -0.03366, -0.05798],
            [0.00473, -0.02254, 0.02988, -0.16510, -0.00306, 0.08742],
            [0.06210, -0.06509, -0.05339, -0.01710, 0.02091, 0.16012],
            [-0.03420, 0.07774, -0.09774, -0.02604, 0.12584, 0.20872],
        ]]);
        let expected_output_without_init_state = TensorData::from([[
            [0.08679, -0.08776, -0.00528, -0.15969, -0.05322, -0.08863],
            [-0.02577, -0.05057, 0.00033, -0.17558, -0.03679, 0.03142],
            [0.02942, -0.07411, -0.06044, -0.03601, -0.09998, 0.04846],
            [-0.04026, 0.07178, -0.10189, -0.07349, -0.04576, 0.05550],
        ]]);
        let expected_hn_with_init_state = TensorData::from([
            [[-0.03420, 0.07774, -0.09774]],
            [[-0.15635, -0.03366, -0.05798]],
        ]);
        let expected_hn_without_init_state = TensorData::from([
            [[-0.04026, 0.07178, -0.10189]],
            [[-0.15969, -0.05322, -0.08863]],
        ]);


        let (output_with_init_state, state_with_init_state) =
            rnn.forward(input.clone(), Some(RnnState::new(h0)));
        let (output_without_init_state, state_without_init_state) = rnn.forward(input, None);

        let tolerance = Tolerance::permissive();
        output_with_init_state
            .to_data()
            .assert_approx_eq::<FT>(&expected_output_with_init_state, tolerance);
        output_without_init_state
            .to_data()
            .assert_approx_eq::<FT>(&expected_output_without_init_state, tolerance);
        state_with_init_state
            .hidden
            .to_data()
            .assert_approx_eq::<FT>(&expected_hn_with_init_state, tolerance);
        state_without_init_state
            .hidden
            .to_data()
            .assert_approx_eq::<FT>(&expected_hn_without_init_state, tolerance);
    }

    #[test]
    fn display_rnn() {
        let config = RnnConfig::new(2, 3, true);

        let layer = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{layer}"),
            "Rnn {d_input: 2, d_hidden: 3, bias: true, params: 84}"
        );
    }

    #[test]
    fn display_birnn() {
        let config = BiRnnConfig::new(2, 3, true);

        let layer = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{layer}"),
            "BiRnn {d_input: 2, d_hidden: 3, bias: true, params: 168}"
        );
    }
}