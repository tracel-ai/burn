use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::nn::Initializer;
use crate::nn::LinearConfig;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use burn_tensor::activation;

use super::gate_controller::GateController;

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
    input_gate: GateController<B>,
    forget_gate: GateController<B>,
    output_gate: GateController<B>,
    cell_gate: GateController<B>,
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

    /// Initialize a new [lstm](Lstm) module with a [record](LstmRecord).
    pub fn init_with<B: Backend>(&self, record: LstmRecord<B>) -> Lstm<B> {
        let linear_config = LinearConfig {
            d_input: self.d_input,
            d_output: self.d_hidden,
            bias: self.bias,
            initializer: self.initializer.clone(),
        };

        Lstm {
            input_gate: GateController::new_with(&linear_config, record.input_gate),
            forget_gate: GateController::new_with(&linear_config, record.forget_gate),
            output_gate: GateController::new_with(&linear_config, record.output_gate),
            cell_gate: GateController::new_with(&linear_config, record.cell_gate),
            d_hidden: self.d_hidden,
        }
    }
}

impl<B: Backend> Lstm<B> {
    /// Applies the forward pass on the input tensor. This LSTM implementation
    /// returns the cell state and hidden state for each element in a sequence (i.e., across `seq_length`),
    /// producing 3-dimensional tensors where the dimensions represent [batch_size, sequence_length, hidden_size].
    ///
    /// Parameters:
    ///     batched_input: The input tensor of shape [batch_size, sequence_length, input_size].
    ///     state: An optional tuple of tensors representing the initial cell state and hidden state.
    ///            Each state tensor has shape [batch_size, hidden_size].
    ///            If no initial state is provided, these tensors are initialized to zeros.
    ///
    /// Returns:
    ///     A tuple of tensors, where the first tensor represents the cell states and
    ///     the second tensor represents the hidden states for each sequence element.
    ///     Both output tensors have the shape [batch_size, sequence_length, hidden_size].
    pub fn forward(
        &self,
        batched_input: Tensor<B, 3>,
        state: Option<(Tensor<B, 2>, Tensor<B, 2>)>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let device = batched_input.clone().device();
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
        state: Option<(Tensor<B, 2>, Tensor<B, 2>)>,
        batch_size: usize,
        seq_length: usize,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mut batched_cell_state = Tensor::zeros([batch_size, seq_length, self.d_hidden], device);
        let mut batched_hidden_state =
            Tensor::zeros([batch_size, seq_length, self.d_hidden], device);

        let (mut cell_state, mut hidden_state) = match state {
            Some((cell_state, hidden_state)) => (cell_state, hidden_state),
            None => (
                Tensor::zeros([batch_size, self.d_hidden], device),
                Tensor::zeros([batch_size, self.d_hidden], device),
            ),
        };

        for (input_t, t) in input_timestep_iter {
            let input_t = input_t.squeeze(1);
            // f(orget)g(ate) tensors
            let biased_fg_input_sum = self.gate_product(&input_t, &hidden_state, &self.forget_gate);
            let forget_values = activation::sigmoid(biased_fg_input_sum); // to multiply with cell state

            // i(nput)g(ate) tensors
            let biased_ig_input_sum = self.gate_product(&input_t, &hidden_state, &self.input_gate);
            let add_values = activation::sigmoid(biased_ig_input_sum);

            // o(output)g(ate) tensors
            let biased_og_input_sum = self.gate_product(&input_t, &hidden_state, &self.output_gate);
            let output_values = activation::sigmoid(biased_og_input_sum);

            // c(ell)g(ate) tensors
            let biased_cg_input_sum = self.gate_product(&input_t, &hidden_state, &self.cell_gate);
            let candidate_cell_values = biased_cg_input_sum.tanh();

            cell_state = forget_values * cell_state.clone() + add_values * candidate_cell_values;
            hidden_state = output_values * cell_state.clone().tanh();

            let unsqueezed_shape = [cell_state.shape().dims[0], 1, cell_state.shape().dims[1]];

            let unsqueezed_cell_state = cell_state.clone().reshape(unsqueezed_shape);
            let unsqueezed_hidden_state = hidden_state.clone().reshape(unsqueezed_shape);

            // store the state for this timestep
            batched_cell_state = batched_cell_state.slice_assign(
                [0..batch_size, t..(t + 1), 0..self.d_hidden],
                unsqueezed_cell_state.clone(),
            );
            batched_hidden_state = batched_hidden_state.slice_assign(
                [0..batch_size, t..(t + 1), 0..self.d_hidden],
                unsqueezed_hidden_state.clone(),
            );
        }

        (batched_cell_state, batched_hidden_state)
    }

    /// Helper function for performing weighted matrix product for a gate and adds
    /// bias, if any.
    ///
    ///  Mathematically, performs `Wx*X + Wh*H + b`, where:
    ///     Wx = weight matrix for the connection to input vector X
    ///     Wh = weight matrix for the connection to hidden state H
    ///     X = input vector
    ///     H = hidden state
    ///     b = bias terms
    fn gate_product(
        &self,
        input: &Tensor<B, 2>,
        hidden: &Tensor<B, 2>,
        gate: &GateController<B>,
    ) -> Tensor<B, 2> {
        let input_product = input.clone().matmul(gate.input_transform.weight.val());
        let hidden_product = hidden.clone().matmul(gate.hidden_transform.weight.val());

        let input_bias = gate
            .input_transform
            .bias
            .as_ref()
            .map(|bias_param| bias_param.val());
        let hidden_bias = gate
            .hidden_transform
            .bias
            .as_ref()
            .map(|bias_param| bias_param.val());

        match (input_bias, hidden_bias) {
            (Some(input_bias), Some(hidden_bias)) => {
                input_product + input_bias.unsqueeze() + hidden_product + hidden_bias.unsqueeze()
            }
            (Some(input_bias), None) => input_product + input_bias.unsqueeze() + hidden_product,
            (None, Some(hidden_bias)) => input_product + hidden_product + hidden_bias.unsqueeze(),
            (None, None) => input_product + hidden_product,
        }
    }
}

/// The configuration for a [bidirectional lstm](BiLstm) module.
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

/// The Lstm module. This implementation is for bidirectional Lstm.
#[derive(Module, Debug)]
pub struct BiLstm<B: Backend> {
    forward: Lstm<B>,
    reverse: Lstm<B>,
    d_hidden: usize,
}

impl BiLstmConfig {
    /// Initialize a new [bidirectional LSTM](BiLstm) module on an automatically selected device.
    pub fn init_devauto<B: Backend>(&self) -> BiLstm<B> {
        let device = B::Device::default();
        self.init(&device)
    }

    /// Initialize a new [bidirectional LSTM](BiLstm) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BiLstm<B> {
        BiLstm {
            forward: LstmConfig::new(self.d_input, self.d_hidden, self.bias).init(device),
            reverse: LstmConfig::new(self.d_input, self.d_hidden, self.bias).init(device),
            d_hidden: self.d_hidden,
        }
    }

    /// Initialize a new [bidirectional LSTM](BiLstm) module with a [record](BiLstmRecord).
    pub fn init_with<B: Backend>(&self, record: BiLstmRecord<B>) -> BiLstm<B> {
        BiLstm {
            forward: LstmConfig::new(self.d_input, self.d_hidden, self.bias)
                .init_with(record.forward),
            reverse: LstmConfig::new(self.d_input, self.d_hidden, self.bias)
                .init_with(record.reverse),
            d_hidden: self.d_hidden,
        }
    }
}

impl<B: Backend> BiLstm<B> {
    /// Applies the forward pass on the input tensor. This LSTM implementation
    /// returns the cell state and hidden state for each element in a sequence (i.e., across `seq_length`),
    /// producing 3-dimensional tensors where the dimensions represent [batch_size, sequence_length, hidden_size * 2].
    ///
    /// Parameters:
    ///     batched_input: The input tensor of shape [batch_size, sequence_length, input_size].
    ///     state: An optional tuple of tensors representing the initial cell state and hidden state.
    ///            Each state tensor has shape [2, batch_size, hidden_size].
    ///            If no initial state is provided, these tensors are initialized to zeros.
    ///
    /// Returns:
    ///     A tuple of tensors, where the first tensor represents the cell states and
    ///     the second tensor represents the hidden states for each sequence element.
    ///     Both output tensors have the shape [batch_size, sequence_length, hidden_size * 2].
    pub fn forward(
        &self,
        batched_input: Tensor<B, 3>,
        state: Option<(Tensor<B, 3>, Tensor<B, 3>)>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let device = batched_input.clone().device();
        let [batch_size, seq_length, _] = batched_input.shape().dims;

        let (cell_state_forward, hidden_state_forward, cell_state_reverse, hidden_state_reverse) =
            match state {
                Some((cell_state, hidden_state)) => (
                    cell_state
                        .clone()
                        .slice([0..1, 0..batch_size, 0..self.d_hidden])
                        .squeeze(0),
                    hidden_state
                        .clone()
                        .slice([0..1, 0..batch_size, 0..self.d_hidden])
                        .squeeze(0),
                    cell_state
                        .slice([1..2, 0..batch_size, 0..self.d_hidden])
                        .squeeze(0),
                    hidden_state
                        .slice([1..2, 0..batch_size, 0..self.d_hidden])
                        .squeeze(0),
                ),
                None => (
                    Tensor::zeros([batch_size, self.d_hidden], &device),
                    Tensor::zeros([batch_size, self.d_hidden], &device),
                    Tensor::zeros([batch_size, self.d_hidden], &device),
                    Tensor::zeros([batch_size, self.d_hidden], &device),
                ),
            };

        let (batched_cell_state_forward, batched_hidden_state_forward) = self.forward.forward(
            batched_input.clone(),
            Some((cell_state_forward, hidden_state_forward)),
        );

        // reverse direction
        let (batched_cell_state_reverse, batched_hidden_state_reverse) = self.reverse.forward_iter(
            batched_input.iter_dim(1).rev().zip((0..seq_length).rev()),
            Some((cell_state_reverse, hidden_state_reverse)),
            batch_size,
            seq_length,
            &device,
        );

        let batched_cell_state = Tensor::cat(
            [batched_cell_state_forward, batched_cell_state_reverse].to_vec(),
            2,
        );
        let batched_hidden_state = Tensor::cat(
            [batched_hidden_state_forward, batched_hidden_state_reverse].to_vec(),
            2,
        );

        (batched_cell_state, batched_hidden_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{module::Param, nn::LinearRecord, TestBackend};
    use burn_tensor::{Data, Distribution};

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
        let mut lstm = config.init_devauto::<TestBackend>();

        fn create_gate_controller(
            weights: f32,
            biases: f32,
            d_input: usize,
            d_output: usize,
            bias: bool,
            initializer: Initializer,
            device: &<TestBackend as Backend>::Device,
        ) -> GateController<TestBackend> {
            let record = LinearRecord {
                weight: Param::from(Tensor::from_data(Data::from([[weights]]), device)),
                bias: Some(Param::from(Tensor::from_data(Data::from([biases]), device))),
            };
            GateController::create_with_weights(
                d_input,
                d_output,
                bias,
                initializer,
                record.clone(),
                record,
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

        let (cell_state_batch, hidden_state_batch) = lstm.forward(input, None);
        let cell_state = cell_state_batch
            .select(0, Tensor::arange(0..1, &device))
            .squeeze(0);
        let hidden_state = hidden_state_batch
            .select(0, Tensor::arange(0..1, &device))
            .squeeze(0);
        cell_state
            .to_data()
            .assert_approx_eq(&Data::from([[0.046]]), 3);
        hidden_state
            .to_data()
            .assert_approx_eq(&Data::from([[0.024]]), 3)
    }

    #[test]
    fn test_batched_forward_pass() {
        let device = Default::default();
        let lstm = LstmConfig::new(64, 1024, true).init::<TestBackend>(&device);
        let batched_input =
            Tensor::<TestBackend, 3>::random([8, 10, 64], Distribution::Default, &device);

        let (cell_state, hidden_state) = lstm.forward(batched_input, None);

        assert_eq!(cell_state.shape().dims, [8, 10, 1024]);
        assert_eq!(hidden_state.shape().dims, [8, 10, 1024]);
    }

    #[test]
    fn test_bidirectional() {
        TestBackend::seed(0);
        let config = BiLstmConfig::new(2, 4, true);
        let device = Default::default();
        let mut lstm = config.init::<TestBackend>(&device);

        fn create_gate_controller<const D1: usize, const D2: usize>(
            input_weights: [[f32; D1]; D2],
            input_biases: [f32; D1],
            hidden_weights: [[f32; D1]; D1],
            hidden_biases: [f32; D1],
            device: &<TestBackend as Backend>::Device,
        ) -> GateController<TestBackend> {
            let d_input = input_weights[0].len();
            let d_output = input_weights.len();

            let input_record = LinearRecord {
                weight: Param::from(Tensor::from_data(Data::from(input_weights), device)),
                bias: Some(Param::from(Tensor::from_data(
                    Data::from(input_biases),
                    device,
                ))),
            };
            let hidden_record = LinearRecord {
                weight: Param::from(Tensor::from_data(Data::from(hidden_weights), device)),
                bias: Some(Param::from(Tensor::from_data(
                    Data::from(hidden_biases),
                    device,
                ))),
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
            Data::from([[[-0.131, -1.591], [1.378, -1.867], [0.397, 0.047]]]),
            &device,
        );

        lstm.forward.input_gate = create_gate_controller(
            [[0.078, 0.234, 0.398, 0.333], [0.452, 0.124, -0.042, -0.152]],
            [0.196, 0.094, -0.270, 0.008],
            [
                [0.054, 0.057, 0.282, 0.021],
                [0.065, -0.303, -0.499, 0.069],
                [-0.007, 0.226, -0.131, -0.307],
                [-0.025, 0.072, 0.197, 0.129],
            ],
            [0.278, -0.211, 0.435, -0.162],
            &device,
        );

        lstm.forward.forget_gate = create_gate_controller(
            [
                [-0.187, -0.201, 0.078, -0.314],
                [0.169, 0.229, 0.218, 0.466],
            ],
            [0.320, -0.135, -0.301, 0.180],
            [
                [0.392, -0.028, 0.470, -0.025],
                [-0.284, -0.286, -0.211, -0.001],
                [0.245, -0.259, 0.102, -0.379],
                [-0.096, -0.462, 0.170, 0.232],
            ],
            [0.458, 0.039, 0.287, -0.327],
            &device,
        );

        lstm.forward.cell_gate = create_gate_controller(
            [
                [-0.216, 0.256, 0.369, 0.160],
                [0.453, -0.238, 0.306, -0.411],
            ],
            [0.360, 0.001, 0.303, 0.438],
            [
                [0.356, -0.185, 0.494, 0.325],
                [0.111, -0.388, 0.051, -0.150],
                [-0.434, 0.296, -0.185, 0.290],
                [-0.010, -0.023, 0.460, 0.238],
            ],
            [0.268, -0.136, -0.452, 0.471],
            &device,
        );

        lstm.forward.output_gate = create_gate_controller(
            [[0.235, -0.132, 0.049, 0.157], [-0.280, 0.229, 0.102, 0.448]],
            [0.237, -0.396, -0.134, -0.047],
            [
                [-0.243, 0.196, 0.087, 0.163],
                [0.138, -0.247, -0.401, -0.462],
                [0.030, -0.263, 0.473, 0.259],
                [-0.413, -0.173, -0.206, 0.324],
            ],
            [-0.364, -0.023, 0.215, -0.401],
            &device,
        );

        lstm.reverse.input_gate = create_gate_controller(
            [
                [0.220, -0.191, 0.062, -0.443],
                [-0.112, -0.353, -0.443, 0.080],
            ],
            [-0.418, 0.209, 0.297, -0.429],
            [
                [-0.121, -0.408, 0.132, -0.450],
                [0.231, 0.154, -0.294, 0.022],
                [0.378, 0.239, 0.176, -0.361],
                [0.480, 0.427, -0.156, -0.137],
            ],
            [0.267, -0.474, -0.393, 0.190],
            &device,
        );

        lstm.reverse.forget_gate = create_gate_controller(
            [
                [0.151, 0.148, 0.341, -0.112],
                [-0.368, -0.476, 0.003, 0.083],
            ],
            [-0.489, -0.361, -0.035, 0.328],
            [
                [0.460, -0.124, -0.377, -0.033],
                [-0.296, 0.162, 0.456, -0.271],
                [0.320, 0.235, 0.383, 0.423],
                [-0.167, 0.332, -0.493, 0.086],
            ],
            [-0.425, 0.219, 0.294, -0.075],
            &device,
        );

        lstm.reverse.cell_gate = create_gate_controller(
            [
                [-0.451, 0.285, 0.305, -0.344],
                [-0.399, 0.344, -0.022, 0.263],
            ],
            [0.215, -0.028, 0.097, 0.197],
            [
                [0.072, 0.106, -0.030, 0.056],
                [-0.278, -0.256, -0.129, -0.252],
                [-0.305, 0.219, 0.045, -0.123],
                [0.224, 0.011, -0.199, -0.362],
            ],
            [0.086, 0.466, -0.152, 0.353],
            &device,
        );

        lstm.reverse.output_gate = create_gate_controller(
            [
                [0.057, -0.357, 0.031, 0.235],
                [-0.492, -0.109, -0.316, -0.422],
            ],
            [0.233, 0.053, 0.162, -0.465],
            [
                [0.240, 0.223, -0.188, -0.181],
                [-0.427, -0.390, -0.176, -0.338],
                [-0.158, 0.152, -0.105, 0.106],
                [-0.223, -0.186, -0.059, 0.319],
            ],
            [0.207, 0.295, 0.361, 0.029],
            &device,
        );

        let expected_result = Data::from([[
            [
                -0.01604, 0.02718, -0.14959, 0.10219, 0.34534, 0.06087, 0.07809, 0.01806,
            ],
            [
                -0.13098, 0.07478, -0.10684, 0.15549, 0.19981, 0.12038, 0.19815, -0.02509,
            ],
            [
                0.09250, 0.03285, -0.04502, 0.24134, 0.03017, 0.11454, 0.01943, 0.06517,
            ],
        ]]);

        let (_, hidden_state) = lstm.forward(input, None);

        hidden_state.to_data().assert_approx_eq(&expected_result, 3)
    }
}
