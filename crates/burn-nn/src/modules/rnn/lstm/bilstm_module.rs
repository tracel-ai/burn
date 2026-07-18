use burn_core as burn;

use crate::activation::ActivationConfig;
use crate::{Lstm, LstmConfig, LstmState};
use alloc::vec;
use burn::Tensor;
use burn::config::Config;
use burn::module::{Content, DisplaySettings, Initializer, Module, ModuleDisplay};
use burn::prelude::Device;
use burn::prelude::s;

/// Configuration to create a [BiLstm](BiLstm) module using the [init function](BiLstmConfig::init).
#[derive(Config, Debug)]
pub struct BiLstmConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If a bias should be applied during the BiLstm transformation.
    pub bias: bool,
    /// BiLstm initializer
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    pub initializer: Initializer,
    /// If true, the input tensor is expected to be `[batch_size, seq_length, input_size]`.
    /// If false, the input tensor is expected to be `[seq_length, batch_size, input_size]`.
    #[config(default = true)]
    pub batch_first: bool,
    /// Optional cell state clip threshold.
    pub clip: Option<f64>,
    /// If true, couples the input and forget gates.
    #[config(default = false)]
    pub input_forget: bool,
    /// Activation function for the input, forget, and output gates.
    #[config(default = "ActivationConfig::Sigmoid")]
    pub gate_activation: ActivationConfig,
    /// Activation function for the cell gate (candidate cell state).
    #[config(default = "ActivationConfig::Tanh")]
    pub cell_activation: ActivationConfig,
    /// Activation function applied to the cell state before computing hidden output.
    #[config(default = "ActivationConfig::Tanh")]
    pub hidden_activation: ActivationConfig,
}

/// The BiLstm module. This implementation is for Bidirectional LSTM.
///
/// Introduced in the paper: [Framewise phoneme classification with bidirectional LSTM and other neural network architectures](https://www.cs.toronto.edu/~graves/ijcnn_2005.pdf).
///
/// Should be created with [BiLstmConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct BiLstm {
    /// LSTM for the forward direction.
    pub forward: Lstm,
    /// LSTM for the reverse direction.
    pub reverse: Lstm,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If true, input is `[batch_size, seq_length, input_size]`.
    /// If false, input is `[seq_length, batch_size, input_size]`.
    pub batch_first: bool,
}

impl ModuleDisplay for BiLstm {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_input, _] = self
            .forward
            .input_gate
            .input_transform
            .weight
            .shape()
            .dims();
        let bias = self.forward.input_gate.input_transform.bias.is_some();

        content
            .add("d_input", &d_input)
            .add("d_hidden", &self.d_hidden)
            .add("bias", &bias)
            .optional()
    }
}

impl BiLstmConfig {
    /// Initialize a new [Bidirectional LSTM](BiLstm) module.
    pub fn init(&self, device: &Device) -> BiLstm {
        // Internal LSTMs always use batch_first=true; BiLstm handles layout conversion
        let base_config = LstmConfig::new(self.d_input, self.d_hidden, self.bias)
            .with_initializer(self.initializer.clone())
            .with_batch_first(true)
            .with_clip(self.clip)
            .with_input_forget(self.input_forget)
            .with_gate_activation(self.gate_activation.clone())
            .with_cell_activation(self.cell_activation.clone())
            .with_hidden_activation(self.hidden_activation.clone());

        BiLstm {
            forward: base_config.clone().init(device),
            reverse: base_config.init(device),
            d_hidden: self.d_hidden,
            batch_first: self.batch_first,
        }
    }
}

impl BiLstm {
    /// Applies the forward pass on the input tensor. This Bidirectional LSTM implementation
    /// returns the state for each element in a sequence (i.e., across seq_length) and a final state.
    ///
    /// ## Parameters:
    /// - batched_input: The input tensor of shape:
    ///   - `[batch_size, sequence_length, input_size]` if `batch_first` is true (default)
    ///   - `[sequence_length, batch_size, input_size]` if `batch_first` is false
    /// - state: An optional `LstmState` representing the initial cell state and hidden state.
    ///   Each state tensor has shape `[2, batch_size, hidden_size]`.
    ///   If no initial state is provided, these tensors are initialized to zeros.
    ///
    /// ## Returns:
    /// - output: A tensor represents the output features of LSTM. Shape:
    ///   - `[batch_size, sequence_length, hidden_size * 2]` if `batch_first` is true
    ///   - `[sequence_length, batch_size, hidden_size * 2]` if `batch_first` is false
    /// - state: A `LstmState` represents the final forward and reverse states. Both `state.cell` and
    ///   `state.hidden` have the shape `[2, batch_size, hidden_size]`.
    pub fn forward(
        &self,
        batched_input: Tensor<3>,
        state: Option<LstmState<3>>,
    ) -> (Tensor<3>, LstmState<3>) {
        // Convert to batch-first layout internally if needed
        let batched_input = if self.batch_first {
            batched_input
        } else {
            batched_input.swap_dims(0, 1)
        };

        let device = batched_input.clone().device();
        let [batch_size, seq_length, _] = batched_input.shape().dims();

        let state_forward = state.clone().map(|st| st.slice(s![0]).squeeze_dim(0));
        let state_reverse = state.map(|st| st.slice(s![1]).squeeze_dim(0));

        // forward direction
        let (hidden_forward, state_forward) =
            self.forward.forward(batched_input.clone(), state_forward);

        // reverse direction
        let (hidden_reverse, state_reverse) = self.reverse.forward_iter(
            batched_input.iter_dim(1).enumerate().rev(),
            state_reverse,
            batch_size,
            seq_length,
            &device,
        );

        let output = Tensor::cat([hidden_forward, hidden_reverse].to_vec(), 2);

        // Convert output back to seq-first layout if needed
        let output = if self.batch_first {
            output
        } else {
            output.swap_dims(0, 1)
        };

        let state = LstmState::stack(vec![state_forward, state_reverse], 0);

        (output, state)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{GateController, Linear};
    use burn_core::module::Param;
    use burn_core::prelude::Device;
    use burn_core::tensor::{TensorData, Tolerance};

    type FT = f32;

    #[test]
    fn display_bilstm() {
        let config = BiLstmConfig::new(2, 3, true);

        let layer = config.init(&Default::default());

        assert_eq!(
            alloc::format!("{layer}"),
            "BiLstm {d_input: 2, d_hidden: 3, bias: true, params: 168}"
        );
    }

    #[test]
    fn test_bidirectional() {
        let device = Device::default();
        device.seed(0);

        let config = BiLstmConfig::new(2, 3, true);
        let mut lstm = config.init(&device);

        fn create_gate_controller<const D1: usize, const D2: usize>(
            input_weights: [[f32; D1]; D2],
            input_biases: [f32; D1],
            hidden_weights: [[f32; D1]; D1],
            hidden_biases: [f32; D1],
            device: &Device,
        ) -> GateController {
            let input_record = Linear {
                weight: Param::from_data(TensorData::from(input_weights), device),
                bias: Some(Param::from_data(TensorData::from(input_biases), device)),
            };
            let hidden_record = Linear {
                weight: Param::from_data(TensorData::from(hidden_weights), device),
                bias: Some(Param::from_data(TensorData::from(hidden_biases), device)),
            };
            GateController::create_with_weights(input_record, hidden_record)
        }

        let input = Tensor::<3>::from_data(
            TensorData::from([[
                [0.949, -0.861],
                [0.892, 0.927],
                [-0.173, -0.301],
                [-0.081, 0.992],
            ]]),
            &device,
        );
        let h0 = Tensor::<3>::from_data(
            TensorData::from([[[0.280, 0.360, -1.242]], [[-0.588, 0.729, -0.788]]]),
            &device,
        );
        let c0 = Tensor::<3>::from_data(
            TensorData::from([[[0.723, 0.397, -0.262]], [[0.471, 0.613, 1.885]]]),
            &device,
        );

        lstm.forward.input_gate = create_gate_controller(
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

        lstm.forward.forget_gate = create_gate_controller(
            [[-0.342, -0.084, -0.420], [-0.432, 0.119, 0.191]],
            [0.315, -0.413, -0.041],
            [
                [0.453, 0.063, 0.561],
                [0.211, 0.149, 0.213],
                [-0.499, -0.158, 0.068],
            ],
            [-0.431, -0.535, 0.125],
            &device,
        );

        lstm.forward.cell_gate = create_gate_controller(
            [[-0.046, -0.382, 0.321], [-0.533, 0.558, 0.004]],
            [-0.358, 0.282, -0.078],
            [
                [-0.358, 0.109, 0.139],
                [-0.345, 0.091, -0.368],
                [-0.508, 0.221, -0.507],
            ],
            [0.502, -0.509, -0.247],
            &device,
        );

        lstm.forward.output_gate = create_gate_controller(
            [[-0.577, -0.359, 0.216], [-0.550, 0.268, 0.243]],
            [-0.227, -0.274, 0.039],
            [
                [-0.383, 0.449, 0.222],
                [-0.357, -0.093, 0.449],
                [-0.106, 0.236, 0.360],
            ],
            [-0.361, -0.209, -0.454],
            &device,
        );

        lstm.reverse.input_gate = create_gate_controller(
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

        lstm.reverse.forget_gate = create_gate_controller(
            [[-0.154, -0.432, -0.547], [-0.369, -0.310, -0.175]],
            [0.141, 0.004, 0.055],
            [
                [-0.005, -0.277, -0.515],
                [-0.011, -0.101, -0.365],
                [0.426, 0.379, 0.337],
            ],
            [-0.382, 0.331, -0.176],
            &device,
        );

        lstm.reverse.cell_gate = create_gate_controller(
            [[-0.571, 0.228, -0.287], [-0.331, 0.110, 0.219]],
            [-0.206, -0.546, 0.462],
            [
                [0.449, -0.240, 0.071],
                [-0.045, 0.131, 0.124],
                [0.138, -0.201, 0.191],
            ],
            [-0.030, 0.211, -0.352],
            &device,
        );

        lstm.reverse.output_gate = create_gate_controller(
            [[0.491, -0.442, 0.333], [0.313, -0.121, -0.070]],
            [-0.387, -0.250, 0.066],
            [
                [-0.030, 0.268, 0.299],
                [-0.019, -0.280, -0.314],
                [0.466, -0.365, -0.248],
            ],
            [-0.398, -0.199, -0.566],
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
        let expected_cn_with_init_state = TensorData::from([
            [[-0.13593, 0.17125, -0.22395]],
            [[-0.45425, -0.11206, -0.12908]],
        ]);
        let expected_hn_without_init_state = TensorData::from([
            [[-0.04026, 0.07178, -0.10189]],
            [[-0.15969, -0.05322, -0.08863]],
        ]);
        let expected_cn_without_init_state = TensorData::from([
            [[-0.15839, 0.15923, -0.23569]],
            [[-0.47407, -0.17493, -0.19643]],
        ]);

        let (output_with_init_state, state_with_init_state) =
            lstm.forward(input.clone(), Some(LstmState::new(c0, h0)));
        let (output_without_init_state, state_without_init_state) = lstm.forward(input, None);

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
        state_with_init_state
            .cell
            .to_data()
            .assert_approx_eq::<FT>(&expected_cn_with_init_state, tolerance);
        state_without_init_state
            .hidden
            .to_data()
            .assert_approx_eq::<FT>(&expected_hn_without_init_state, tolerance);
        state_without_init_state
            .cell
            .to_data()
            .assert_approx_eq::<FT>(&expected_cn_without_init_state, tolerance);
    }
}
