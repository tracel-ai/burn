use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::nn::rnn::gate_controller::GateController;
use crate::nn::Initializer;
use crate::tensor::activation;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

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

/// Configuration to create a [Lstm](Lstm) module using the [init function](LstmConfig::init).
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
///
/// Introduced in the paper: [Long Short-Term Memory](https://www.researchgate.net/publication/13853244).
///
/// Should be created with [LstmConfig].
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
    /// returns the state for each element in a sequence (i.e., across seq_length) and a final state.
    ///
    /// ## Parameters:
    /// - batched_input: The input tensor of shape `[batch_size, sequence_length, input_size]`.
    /// - state: An optional `LstmState` representing the initial cell state and hidden state.
    ///          Each state tensor has shape `[batch_size, hidden_size]`.
    ///          If no initial state is provided, these tensors are initialized to zeros.
    ///
    /// ## Returns:
    /// - output: A tensor represents the output features of LSTM. Shape: `[batch_size, sequence_length, hidden_size]`
    /// - state: A `LstmState` represents the final states. Both `state.cell` and `state.hidden` have the shape
    ///          `[batch_size, hidden_size]`.
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

/// Configuration to create a [BiLstm](BiLstm) module using the [init function](BiLstmConfig::init).
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
///
/// Introduced in the paper: [Framewise phoneme classification with bidirectional LSTM and other neural network architectures](https://www.cs.toronto.edu/~graves/ijcnn_2005.pdf).
///
/// Should be created with [BiLstmConfig].
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
    /// returns the state for each element in a sequence (i.e., across seq_length) and a final state.
    ///
    /// ## Parameters:
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
    use crate::tensor::{Data, Device, Distribution};
    use crate::{module::Param, nn::LinearRecord, TestBackend};

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

        assert_eq!(output.dims(), [1, 2, 1024]);
        assert_eq!(state.cell.dims(), [1, 1024]);
        assert_eq!(state.hidden.dims(), [1, 1024]);
    }

    #[test]
    #[cfg(feature = "std")]
    fn test_batched_backward_pass() {
        use crate::tensor::Shape;
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
            Data::from([[
                [0.949, -0.861],
                [0.892, 0.927],
                [-0.173, -0.301],
                [-0.081, 0.992],
            ]]),
            &device,
        );
        let h0 = Tensor::<TestBackend, 3>::from_data(
            Data::from([[[0.280, 0.360, -1.242]], [[-0.588, 0.729, -0.788]]]),
            &device,
        );
        let c0 = Tensor::<TestBackend, 3>::from_data(
            Data::from([[[0.723, 0.397, -0.262]], [[0.471, 0.613, 1.885]]]),
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

        let expected_output_with_init_state = Data::from([[
            [0.23764, -0.03442, 0.04414, -0.15635, -0.03366, -0.05798],
            [0.00473, -0.02254, 0.02988, -0.16510, -0.00306, 0.08742],
            [0.06210, -0.06509, -0.05339, -0.01710, 0.02091, 0.16012],
            [-0.03420, 0.07774, -0.09774, -0.02604, 0.12584, 0.20872],
        ]]);
        let expected_output_without_init_state = Data::from([[
            [0.08679, -0.08776, -0.00528, -0.15969, -0.05322, -0.08863],
            [-0.02577, -0.05057, 0.00033, -0.17558, -0.03679, 0.03142],
            [0.02942, -0.07411, -0.06044, -0.03601, -0.09998, 0.04846],
            [-0.04026, 0.07178, -0.10189, -0.07349, -0.04576, 0.05550],
        ]]);
        let expected_hn_with_init_state = Data::from([
            [[-0.03420, 0.07774, -0.09774]],
            [[-0.15635, -0.03366, -0.05798]],
        ]);
        let expected_cn_with_init_state = Data::from([
            [[-0.13593, 0.17125, -0.22395]],
            [[-0.45425, -0.11206, -0.12908]],
        ]);
        let expected_hn_without_init_state = Data::from([
            [[-0.04026, 0.07178, -0.10189]],
            [[-0.15969, -0.05322, -0.08863]],
        ]);
        let expected_cn_without_init_state = Data::from([
            [[-0.15839, 0.15923, -0.23569]],
            [[-0.47407, -0.17493, -0.19643]],
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
