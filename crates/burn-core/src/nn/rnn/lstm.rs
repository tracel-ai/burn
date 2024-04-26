use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::nn::rnn::gate_controller::GateController;
use crate::nn::Initializer;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use burn_tensor::activation;

/// A LstmState is used to store cell state and hidden state in LSTM.
pub struct LstmState<B: Backend, const D: usize> {
    /// The cell state.
    pub cell: Tensor<B, D>,
    /// The hidden state.
    pub hidden: Tensor<B, D>,
}

impl<B: Backend, const D: usize> LstmState<B, D> {
    /// Initialize a new [LSTM State](LstmState).
    pub fn new(cell: Tensor<B, D>, hidden: Tensor<B, D>) -> Self {
        Self { cell, hidden }
    }
}

/// The configuration for a [lstm](Lstm) module.
#[derive(Config)]
pub struct LstmConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If a bias should be applied during the Lstm transformation.
    pub bias: bool,
    /// Lstm initializer
    #[config(default = "Initializer::XavierNormal{gain:1.0}")]
    pub initializer: Initializer,
}

/// The Lstm module. This implementation is for a unidirectional, stateless, Lstm.
#[derive(Module, Debug)]
pub struct Lstm<B: Backend> {
    /// The input gate regulates which information to update and store in the cell state at each time step.
    pub input_gate: GateController<B>,
    /// The forget gate is used to control which information to discard or keep in the memory cell at each time step.
    pub forget_gate: GateController<B>,
    /// The output gate determines which information from the cell state to output at each time step.
    pub output_gate: GateController<B>,
    /// The cell gate is used to compute the cell state that stores and carries information through time.
    pub cell_gate: GateController<B>,
    d_hidden: usize,
}

impl LstmConfig {
    /// Initialize a new [lstm](Lstm) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Lstm<B> {
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

        Lstm {
            input_gate: new_gate(),
            forget_gate: new_gate(),
            output_gate: new_gate(),
            cell_gate: new_gate(),
            d_hidden: self.d_hidden,
        }
    }
}

impl<B: Backend> Lstm<B> {
    /// Applies the forward pass on the input tensor. This LSTM implementation
    /// returns hidden state for each element in a sequence (i.e., across `seq_length`) and a final state,
    /// producing 3-dimensional tensors where the dimensions represent `[batch_size, sequence_length, hidden_size]`.
    ///
    /// ## Parameters:
    /// - batched_input: The input tensor of shape `[batch_size, sequence_length, input_size]`.
    /// - state: An optional `LstmState` representing the initial cell state and hidden state.
    ///          Each state tensor has shape `[batch_size, hidden_size]`.
    ///          If no initial state is provided, these tensors are initialized to zeros.
    ///
    /// ## Returns:
    /// - output: A tensor represents the output features of LSTM. Shape: `[batch_size, sequence_length, hidden_size]`
    /// - state: A `LstmState` represents the final forward and reverse states. Both `state.cell` and
    ///          `state.hidden` have the shape `[batch_size, hidden_size]`.
    pub fn forward(
        &self,
        batched_input: Tensor<B, 3>,
        state: Option<LstmState<B, 2>>,
    ) -> (Tensor<B, 3>, LstmState<B, 2>) {
        let device = batched_input.device();
        let [batch_size, seq_length, _] = batched_input.dims();

        self.forward_iter(
            batched_input.iter_dim(1).zip(0..seq_length),
            state,
            batch_size,
            seq_length,
            &device,
        )
    }

    fn forward_iter<I: Iterator<Item = (Tensor<B, 3>, usize)>>(
        &self,
        input_timestep_iter: I,
        state: Option<LstmState<B, 2>>,
        batch_size: usize,
        seq_length: usize,
        device: &B::Device,
    ) -> (Tensor<B, 3>, LstmState<B, 2>) {
        let mut batched_hidden_state =
            Tensor::empty([batch_size, seq_length, self.d_hidden], device);

        let (mut cell_state, mut hidden_state) = match state {
            Some(state) => (state.cell, state.hidden),
            None => (
                Tensor::zeros([batch_size, self.d_hidden], device),
                Tensor::zeros([batch_size, self.d_hidden], device),
            ),
        };

        for (input_t, t) in input_timestep_iter {
            let input_t = input_t.squeeze(1);
            // f(orget)g(ate) tensors
            let biased_fg_input_sum = self
                .forget_gate
                .gate_product(input_t.clone(), hidden_state.clone());
            let forget_values = activation::sigmoid(biased_fg_input_sum); // to multiply with cell state

            // i(nput)g(ate) tensors
            let biased_ig_input_sum = self
                .input_gate
                .gate_product(input_t.clone(), hidden_state.clone());
            let add_values = activation::sigmoid(biased_ig_input_sum);

            // o(output)g(ate) tensors
            let biased_og_input_sum = self
                .output_gate
                .gate_product(input_t.clone(), hidden_state.clone());
            let output_values = activation::sigmoid(biased_og_input_sum);

            // c(ell)g(ate) tensors
            let biased_cg_input_sum = self
                .cell_gate
                .gate_product(input_t.clone(), hidden_state.clone());
            let candidate_cell_values = biased_cg_input_sum.tanh();

            cell_state = forget_values * cell_state.clone() + add_values * candidate_cell_values;
            hidden_state = output_values * cell_state.clone().tanh();

            let unsqueezed_hidden_state = hidden_state.clone().unsqueeze_dim(1);

            // store the hidden state for this timestep
            batched_hidden_state = batched_hidden_state.slice_assign(
                [0..batch_size, t..(t + 1), 0..self.d_hidden],
                unsqueezed_hidden_state.clone(),
            );
        }

        (
            batched_hidden_state,
            LstmState::new(cell_state, hidden_state),
        )
    }
}

/// The configuration for a [Bidirectional LSTM](BiLstm) module.
#[derive(Config)]
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
}

/// The BiLstm module. This implementation is for Bidirectional LSTM.
#[derive(Module, Debug)]
pub struct BiLstm<B: Backend> {
    /// LSTM for the forward direction.
    pub forward: Lstm<B>,
    /// LSTM for the reverse direction.
    pub reverse: Lstm<B>,
    d_hidden: usize,
}

impl BiLstmConfig {
    /// Initialize a new [Bidirectional LSTM](BiLstm) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BiLstm<B> {
        BiLstm {
            forward: LstmConfig::new(self.d_input, self.d_hidden, self.bias)
                .with_initializer(self.initializer.clone())
                .init(device),
            reverse: LstmConfig::new(self.d_input, self.d_hidden, self.bias)
                .with_initializer(self.initializer.clone())
                .init(device),
            d_hidden: self.d_hidden,
        }
    }
}

impl<B: Backend> BiLstm<B> {
    /// Applies the forward pass on the input tensor. This Bidirectional LSTM implementation
    /// returns hidden state for each element in a sequence (i.e., across `seq_length`) and a final state,
    /// producing 3-dimensional tensors where the dimensions represent `[batch_size, sequence_length, hidden_size * 2]`.
    ///
    /// ## Parameters:
    ///
    /// - batched_input: The input tensor of shape `[batch_size, sequence_length, input_size]`.
    /// - state: An optional `LstmState` representing the initial cell state and hidden state.
    ///          Each state tensor has shape `[2, batch_size, hidden_size]`.
    ///          If no initial state is provided, these tensors are initialized to zeros.
    ///
    /// ## Returns:
    /// - output: A tensor represents the output features of LSTM. Shape: `[batch_size, sequence_length, hidden_size * 2]`
    /// - state: A `LstmState` represents the final forward and reverse states. Both `state.cell` and
    ///          `state.hidden` have the shape `[2, batch_size, hidden_size]`.
    pub fn forward(
        &self,
        batched_input: Tensor<B, 3>,
        state: Option<LstmState<B, 3>>,
    ) -> (Tensor<B, 3>, LstmState<B, 3>) {
        let device = batched_input.clone().device();
        let [batch_size, seq_length, _] = batched_input.shape().dims;

        let [init_state_forward, init_state_reverse] = match state {
            Some(state) => {
                let cell_state_forward = state
                    .cell
                    .clone()
                    .slice([0..1, 0..batch_size, 0..self.d_hidden])
                    .squeeze(0);
                let hidden_state_forward = state
                    .hidden
                    .clone()
                    .slice([0..1, 0..batch_size, 0..self.d_hidden])
                    .squeeze(0);
                let cell_state_reverse = state
                    .cell
                    .slice([1..2, 0..batch_size, 0..self.d_hidden])
                    .squeeze(0);
                let hidden_state_reverse = state
                    .hidden
                    .slice([1..2, 0..batch_size, 0..self.d_hidden])
                    .squeeze(0);

                [
                    Some(LstmState::new(cell_state_forward, hidden_state_forward)),
                    Some(LstmState::new(cell_state_reverse, hidden_state_reverse)),
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

        let state = LstmState::new(
            Tensor::stack(
                [final_state_forward.cell, final_state_reverse.cell].to_vec(),
                0,
            ),
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
    use crate::{module::Param, nn::LinearRecord, TestBackend};
    use burn_tensor::{Data, Device, Distribution};

    #[cfg(feature = "std")]
    use crate::TestAutodiffBackend;

    #[test]
    fn test_with_uniform_initializer() {
        TestBackend::seed(0);

        let config = LstmConfig::new(5, 5, false)
            .with_initializer(Initializer::Uniform { min: 0.0, max: 1.0 });
        let lstm = config.init::<TestBackend>(&Default::default());

        let gate_to_data =
            |gate: GateController<TestBackend>| gate.input_transform.weight.val().to_data();

        gate_to_data(lstm.input_gate).assert_within_range(0..1);
        gate_to_data(lstm.forget_gate).assert_within_range(0..1);
        gate_to_data(lstm.output_gate).assert_within_range(0..1);
        gate_to_data(lstm.cell_gate).assert_within_range(0..1);
    }

    /// Test forward pass with simple input vector.
    ///
    /// f_t = sigmoid(0.7*0.1 + 0.7*0) = sigmoid(0.07) = 0.5173928
    /// i_t = sigmoid(0.5*0.1 + 0.5*0) = sigmoid(0.05) = 0.5123725
    /// o_t = sigmoid(1.1*0.1 + 1.1*0) = sigmoid(0.11) = 0.5274723
    /// c_t = tanh(0.9*0.1 + 0.9*0) = tanh(0.09) = 0.0892937

    /// C_t = f_t * 0 + i_t * c_t = 0 + 0.5123725 * 0.0892937 = 0.04575243
    /// h_t = o_t * tanh(C_t) = 0.5274723 * tanh(0.04575243) = 0.5274723 * 0.04568173 = 0.024083648
    #[test]
    fn test_forward_single_input_single_feature() {
        TestBackend::seed(0);
        let config = LstmConfig::new(1, 1, false);
        let device = Default::default();
        let mut lstm = config.init::<TestBackend>(&device);

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
                weight: Param::from_data(Data::from([[weights]]), device),
                bias: Some(Param::from_data(Data::from([biases]), device)),
            };
            let record_2 = LinearRecord {
                weight: Param::from_data(Data::from([[weights]]), device),
                bias: Some(Param::from_data(Data::from([biases]), device)),
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

        lstm.input_gate = create_gate_controller(
            0.5,
            0.0,
            1,
            1,
            false,
            Initializer::XavierUniform { gain: 1.0 },
            &device,
        );
        lstm.forget_gate = create_gate_controller(
            0.7,
            0.0,
            1,
            1,
            false,
            Initializer::XavierUniform { gain: 1.0 },
            &device,
        );
        lstm.cell_gate = create_gate_controller(
            0.9,
            0.0,
            1,
            1,
            false,
            Initializer::XavierUniform { gain: 1.0 },
            &device,
        );
        lstm.output_gate = create_gate_controller(
            1.1,
            0.0,
            1,
            1,
            false,
            Initializer::XavierUniform { gain: 1.0 },
            &device,
        );

        // single timestep with single feature
        let input = Tensor::<TestBackend, 3>::from_data(Data::from([[[0.1]]]), &device);

        // let (cell_state_batch, hidden_state_batch) = lstm.forward(input, None);
        let (output, state) = lstm.forward(input, None);
        state
            .cell
            .to_data()
            .assert_approx_eq(&Data::from([[0.046]]), 3);
        state
            .hidden
            .to_data()
            .assert_approx_eq(&Data::from([[0.024]]), 3);
        output
            .select(0, Tensor::arange(0..1, &device))
            .squeeze(0)
            .to_data()
            .assert_approx_eq(&state.hidden.to_data(), 3);
    }

    #[test]
    fn test_batched_forward_pass() {
        let device = Default::default();
        let lstm = LstmConfig::new(64, 1024, true).init(&device);
        let batched_input =
            Tensor::<TestBackend, 3>::random([8, 10, 64], Distribution::Default, &device);

        let (output, state) = lstm.forward(batched_input, None);

        assert_eq!(output.dims(), [8, 10, 1024]);
        assert_eq!(state.cell.dims(), [8, 1024]);
        assert_eq!(state.hidden.dims(), [8, 1024]);
    }

    #[test]
    fn test_batched_forward_pass_batch_of_one() {
        let device = Default::default();
        let lstm = LstmConfig::new(64, 1024, true).init(&device);
        let batched_input =
            Tensor::<TestBackend, 3>::random([1, 2, 64], Distribution::Default, &device);

        let (output, state) = lstm.forward(batched_input, None);

        assert_eq!(output.dims(), [1, 10, 1024]);
        assert_eq!(state.cell.dims(), [1, 1024]);
        assert_eq!(state.hidden.dims(), [1, 1024]);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_batched_backward_pass() {
        use burn_tensor::Shape;
        let device = Default::default();
        let lstm = LstmConfig::new(64, 32, true).init(&device);
        let shape: Shape<3> = [8, 10, 64].into();
        let batched_input =
            Tensor::<TestAutodiffBackend, 3>::random(shape, Distribution::Default, &device);

        let (output, _) = lstm.forward(batched_input.clone(), None);
        let fake_loss = output;
        let grads = fake_loss.backward();

        let some_gradient = lstm
            .output_gate
            .hidden_transform
            .weight
            .grad(&grads)
            .unwrap();

        // Asserts that the gradients exist and are non-zero
        assert!(*some_gradient.any().into_data().value.first().unwrap());
    }

    #[test]
    fn test_bidirectional() {
        TestBackend::seed(0);
        let config = BiLstmConfig::new(2, 3, true);
        let device = Default::default();
        let mut lstm = config.init(&device);

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
                weight: Param::from_data(Data::from(input_weights), device),
                bias: Some(Param::from_data(Data::from(input_biases), device)),
            };
            let hidden_record = LinearRecord {
                weight: Param::from_data(Data::from(hidden_weights), device),
                bias: Some(Param::from_data(Data::from(hidden_biases), device)),
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
            Data::from([
                [
                    [1.647, -0.499],
                    [-1.991, 0.439],
                    [0.571, 0.563],
                    [0.149, -1.048],
                ],
                [
                    [0.039, -0.786],
                    [-0.703, 1.071],
                    [-0.417, -1.480],
                    [-0.621, -0.827],
                ],
            ]),
            &device,
        );
        let h0 = Tensor::<TestBackend, 3>::from_data(
            Data::from([
                [[0.680, -0.813, 0.760], [0.336, 0.827, -0.749]],
                [[-1.736, -0.235, 0.925], [-0.048, 0.218, 0.909]],
            ]),
            &device,
        );
        let c0 = Tensor::<TestBackend, 3>::from_data(
            Data::from([
                [[-0.298, 0.507, -0.058], [-1.805, 0.768, 0.523]],
                [[0.364, -1.398, 1.188], [0.087, -0.555, 0.500]],
            ]),
            &device,
        );

        lstm.forward.input_gate = create_gate_controller(
            [[0.050, 0.292, -0.044], [-0.392, 0.409, -0.110]],
            [-0.007, 0.483, 0.038],
            [
                [0.036, 0.511, -0.236],
                [-0.232, 0.449, 0.146],
                [-0.282, -0.365, -0.329],
            ],
            [0.355, -0.259, -0.300],
            &device,
        );

        lstm.forward.forget_gate = create_gate_controller(
            [[0.360, -0.228, -0.036], [0.123, -0.077, -0.341]],
            [-0.306, -0.335, -0.039],
            [
                [0.156, -0.156, -0.360],
                [-0.117, -0.429, -0.259],
                [0.023, 0.226, 0.455],
            ],
            [0.255, -0.067, -0.125],
            &device,
        );

        lstm.forward.cell_gate = create_gate_controller(
            [[-0.375, -0.128, 0.363], [0.041, -0.109, 0.071]],
            [0.014, 0.489, 0.218],
            [
                [0.559, -0.561, -0.426],
                [0.205, -0.492, 0.010],
                [0.280, -0.496, -0.220],
            ],
            [0.239, 0.166, -0.176],
            &device,
        );

        lstm.forward.output_gate = create_gate_controller(
            [[0.352, 0.206, 0.020], [0.343, -0.327, 0.208]],
            [-0.451, 0.071, -0.232],
            [
                [-0.257, -0.346, -0.343],
                [0.490, -0.473, 0.208],
                [0.457, 0.105, 0.093],
            ],
            [-0.531, 0.178, -0.475],
            &device,
        );

        lstm.reverse.input_gate = create_gate_controller(
            [[0.098, 0.072, 0.429], [0.397, 0.479, -0.320]],
            [-0.129, 0.442, -0.044],
            [
                [-0.543, 0.344, -0.013],
                [-0.388, 0.389, -0.480],
                [-0.496, -0.193, -0.169],
            ],
            [-0.042, 0.576, -0.465],
            &device,
        );

        lstm.reverse.forget_gate = create_gate_controller(
            [[-0.514, -0.553, -0.569], [-0.045, 0.367, 0.521]],
            [0.240, -0.500, 0.502],
            [
                [0.270, 0.027, 0.411],
                [-0.123, -0.447, -0.051],
                [-0.280, -0.056, 0.261],
            ],
            [0.189, -0.567, 0.117],
            &device,
        );

        lstm.reverse.cell_gate = create_gate_controller(
            [[-0.488, 0.185, -0.163], [-0.243, -0.307, -0.098]],
            [0.368, -0.306, -0.524],
            [
                [0.572, -0.365, 0.348],
                [-0.492, 0.512, -0.023],
                [-0.144, 0.050, 0.098],
            ],
            [0.148, 0.163, -0.546],
            &device,
        );

        lstm.reverse.output_gate = create_gate_controller(
            [[-0.069, -0.455, 0.461], [-0.274, 0.266, 0.519]],
            [0.388, 0.545, -0.388],
            [
                [0.180, -0.462, 0.106],
                [0.543, 0.295, -0.411],
                [-0.011, -0.066, 0.470],
            ],
            [-0.179, -0.196, -0.067],
            &device,
        );

        let expected_output_with_init_state = Data::from([
            [
                [-0.05291, 0.20481, 0.00247, 0.14330, -0.01617, -0.32437],
                [0.06333, 0.17128, -0.08646, 0.23891, -0.39256, -0.08803],
                [0.08696, 0.21229, 0.00791, 0.02564, -0.08598, -0.15525],
                [0.06240, 0.25245, 0.00132, 0.00171, 0.01233, 0.01195],
            ],
            [
                [-0.11673, 0.23612, 0.05902, 0.34088, -0.09401, -0.16047],
                [-0.01053, 0.20343, 0.01439, 0.26776, -0.29267, -0.15661],
                [0.04320, 0.28468, -0.02198, 0.24269, 0.04973, -0.04563],
                [0.07891, 0.24718, -0.04706, 0.13683, -0.01629, 0.03767],
            ],
        ]);
        let expected_output_without_init_state = Data::from([
            [
                [-0.08461, 0.18986, 0.07192, 0.18021, -0.02266, -0.35150],
                [0.06048, 0.17062, -0.05256, 0.29482, -0.40167, -0.12416],
                [0.08438, 0.20755, 0.02044, 0.10186, -0.08353, -0.25673],
                [0.06173, 0.24971, 0.00638, 0.13258, 0.06368, -0.09722],
            ],
            [
                [0.02993, 0.18217, 0.00005, 0.35562, -0.09828, -0.17992],
                [0.09557, 0.16621, -0.02360, 0.28457, -0.29604, -0.21862],
                [0.06623, 0.26088, -0.03991, 0.27286, 0.05034, -0.08039],
                [0.08877, 0.24112, -0.05770, 0.16840, -0.00154, -0.06161],
            ],
        ]);
        let expected_hn_with_init_state = Data::from([
            [[0.06240, 0.25245, 0.00132], [0.07891, 0.24718, -0.04706]],
            [[0.14330, -0.01617, -0.32437], [0.34088, -0.09401, -0.16047]],
        ]);
        let expected_cn_with_init_state = Data::from([
            [[0.27726, 0.43163, 0.00460], [0.40963, 0.47434, -0.15836]],
            [[0.28537, -0.05057, -0.68145], [0.67802, -0.19816, -0.55872]],
        ]);
        let expected_hn_without_init_state = Data::from([
            [[0.06173, 0.24971, 0.00638], [0.08877, 0.24112, -0.05770]],
            [[0.18021, -0.02266, -0.35150], [0.35562, -0.09828, -0.17992]],
        ]);
        let expected_cn_without_init_state = Data::from([
            [[0.27319, 0.42555, 0.02218], [0.47942, 0.46064, -0.19706]],
            [[0.36375, -0.07220, -0.76521], [0.71734, -0.20792, -0.66048]],
        ]);

        let (output_with_init_state, state_with_init_state) =
            lstm.forward(input.clone(), Some(LstmState::new(c0, h0)));
        let (output_without_init_state, state_without_init_state) = lstm.forward(input, None);

        output_with_init_state
            .to_data()
            .assert_approx_eq(&expected_output_with_init_state, 3);
        output_without_init_state
            .to_data()
            .assert_approx_eq(&expected_output_without_init_state, 3);
        state_with_init_state
            .hidden
            .to_data()
            .assert_approx_eq(&expected_hn_with_init_state, 3);
        state_with_init_state
            .cell
            .to_data()
            .assert_approx_eq(&expected_cn_with_init_state, 3);
        state_without_init_state
            .hidden
            .to_data()
            .assert_approx_eq(&expected_hn_without_init_state, 3);
        state_without_init_state
            .cell
            .to_data()
            .assert_approx_eq(&expected_cn_without_init_state, 3);
    }
}
