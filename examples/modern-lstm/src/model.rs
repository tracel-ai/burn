use burn::{
    nn::{
        Dropout, DropoutConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        LstmState, Sigmoid, Tanh,
    },
    prelude::*,
};

/// LSTM Cell implementation with layer normalization.
///
/// Mathematical formulation of LSTM:
/// f_t = σ(W_f · [h_{t-1}, x_t] + b_f)      # Forget gate
/// i_t = σ(W_i · [h_{t-1}, x_t] + b_i]      # Input gate
/// g_t = tanh(W_g · [h_{t-1}, x_t] + b_g]   # Candidate cell state
/// o_t = σ(W_o · [h_{t-1}, x_t] + b_o)      # Output gate
///
/// c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t            # New cell state
/// h_t = o_t ⊙ tanh(c_t)                       # New hidden state
///
/// where:
/// - σ is the sigmoid function
/// - ⊙ is the element-wise multiplication
/// - [h_{t-1}, x_t] represents concatenation

#[derive(Module, Debug)]
pub struct LstmCell<B: Backend> {
    pub hidden_size: usize,
    // Combined weight matrices for efficiency
    // weight_ih layer uses combined weights for [i_t, f_t, g_t, o_t] for input x_t
    // weight_hh layer uses combined weights for [i_t, f_t, g_t, o_t] for hidden state h_{t-1}
    pub weight_ih: Linear<B>,
    pub weight_hh: Linear<B>,
    // Layer Normalization for better training stability. Don't use BatchNorm because the input distribution is always changing for LSTM.
    pub norm_x: LayerNorm<B>, // Normalize gate pre-activations
    pub norm_h: LayerNorm<B>, // Normalize hidden state
    pub norm_c: LayerNorm<B>, // Normalize cell state
    pub dropout: Dropout,
}

/// Configuration to create a Lstm module using the init function.
#[derive(Config, Debug)]
pub struct LstmCellConfig {
    // The size of the input features
    pub input_size: usize,
    // The size of the hidden state
    pub hidden_size: usize,
    // The number of hidden layers
    pub dropout: f64,
}

impl LstmCellConfig {
    // Initialize parameters using best practices:
    // 1. Orthogonal initialization for better gradient flow (here we use Xavier because of the lack of Orthogonal in burn)
    // 2. Initialize forget gate bias to 1.0 to prevent forgetting at start of training
    #[allow(clippy::single_range_in_vec_init)]
    pub fn init<B: Backend>(&self, device: &B::Device) -> LstmCell<B> {
        let initializer = Initializer::XavierNormal { gain: 1.0 };
        let init_bias = Tensor::<B, 1>::ones([self.hidden_size], device);

        let mut weight_ih = LinearConfig::new(self.input_size, 4 * self.hidden_size)
            .with_initializer(initializer.clone())
            .init(device);
        // Set forget gate bias to 1.0 (helps with learning long sequences)
        let bias = weight_ih
            .bias
            .clone()
            .unwrap()
            .val()
            .slice_assign([self.hidden_size..2 * self.hidden_size], init_bias.clone());
        weight_ih.bias = weight_ih.bias.map(|p| p.map(|_t| bias));

        let mut weight_hh = LinearConfig::new(self.hidden_size, 4 * self.hidden_size)
            .with_initializer(initializer)
            .init(device);
        let bias = weight_hh
            .bias
            .clone()
            .unwrap()
            .val()
            .slice_assign([self.hidden_size..2 * self.hidden_size], init_bias);
        weight_hh.bias = weight_hh.bias.map(|p| p.map(|_t| bias));

        LstmCell {
            hidden_size: self.hidden_size,
            weight_ih,
            weight_hh,
            norm_x: LayerNormConfig::new(4 * self.hidden_size).init(device),
            norm_h: LayerNormConfig::new(self.hidden_size).init(device),
            norm_c: LayerNormConfig::new(self.hidden_size).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> LstmCell<B> {
    /// Forward pass of LSTM cell.
    /// Args:
    ///     x: Input tensor of shape (batch_size, input_size)
    ///     state: Tuple of (h_{t-1}, c_{t-1}) each of shape (batch_size, hidden_size)
    /// Returns:
    ///  Tuple of (h_t, c_t) representing new hidden and cell states
    pub fn forward(&self, x: Tensor<B, 2>, state: LstmState<B, 2>) -> LstmState<B, 2> {
        let (h_prev, c_prev) = (state.hidden, state.cell);

        // Combined matrix multiplication for all gates
        // Shape: (batch_size, 4 * hidden_size)
        let gates_x = self.weight_ih.forward(x); // Transform input
        let gates_h = self.weight_hh.forward(h_prev); // Transform previous hidden state

        // Apply layer normalization
        let gates_x = self.norm_x.forward(gates_x);
        // Combined gate pre-activations
        let gates = gates_x + gates_h;

        // Split into individual gates
        // Each gate shape: (batch_size, hidden_size)
        let gates = gates.chunk(4, 1);
        let i_gate = gates[0].clone();
        let f_gate = gates[1].clone();
        let g_gate = gates[2].clone();
        let o_gate = gates[3].clone();

        // Apply gate non-linearities
        let i_t = Sigmoid::new().forward(i_gate);
        let f_t = Sigmoid::new().forward(f_gate);
        let g_t = Tanh::new().forward(g_gate);
        let o_t = Sigmoid::new().forward(o_gate);

        // Update cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
        let c_t = f_t * c_prev + i_t * g_t;
        let c_t = self.norm_c.forward(c_t);

        // Update cell state: h_t = o_t ⊙ tanh(c_t)
        let h_t = o_t * Tanh::new().forward(c_t.clone());
        let h_t = self.norm_h.forward(h_t);

        let h_t = self.dropout.forward(h_t);

        LstmState::new(h_t, c_t)
    }

    // Initialize cell state and hidden state if provided or with zeros
    pub fn init_state(&self, batch_size: usize, device: &B::Device) -> LstmState<B, 2> {
        let cell = Tensor::zeros([batch_size, self.hidden_size], device);
        let hidden = Tensor::zeros([batch_size, self.hidden_size], device);

        LstmState::new(cell, hidden)
    }
}

/// Stacked LSTM implementation supporting multiple layers
/// Each layer processes the output of the previous layer
#[derive(Module, Debug)]
pub struct StackedLstm<B: Backend> {
    pub layers: Vec<LstmCell<B>>,
}

#[derive(Config, Debug)]
pub struct StackedLstmConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout: f64,
}

impl StackedLstmConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> StackedLstm<B> {
        let mut layers: Vec<LstmCell<B>> = vec![];
        // Create list of LSTM cells, one for each layer
        for i in 0..self.num_layers {
            if i == 0 {
                if i < self.num_layers - 1 {
                    layers.push(
                        LstmCellConfig::new(self.input_size, self.hidden_size, self.dropout)
                            .init(device),
                    );
                } else {
                    // No dropout on last layer
                    layers.push(
                        LstmCellConfig::new(self.input_size, self.hidden_size, 0.0).init(device),
                    );
                }
            } else if i < self.num_layers - 1 {
                layers.push(
                    LstmCellConfig::new(self.hidden_size, self.hidden_size, self.dropout)
                        .init(device),
                );
            } else {
                // No dropout on last layer
                layers.push(
                    LstmCellConfig::new(self.hidden_size, self.hidden_size, 0.0).init(device),
                );
            }
        }
        StackedLstm { layers }
    }
}

impl<B: Backend> StackedLstm<B> {
    /// Process input sequence through stacked LSTM layers.
    ///
    /// Args:
    ///     x: Input tensor of shape (batch_size, seq_length, input_size)
    ///     states: Optional initial states for each layer
    ///
    /// Returns:
    ///     Tuple of (output, states) where output has shape (batch_size, seq_length, hidden_size)
    ///     and states is a vector of length num_layers, both cell and hidden state in each element have shape (batch_size, hidden_size)
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        states: Option<Vec<LstmState<B, 2>>>,
    ) -> (Tensor<B, 3>, Vec<LstmState<B, 2>>) {
        let [batch_size, seq_length, _] = x.dims();
        let device = x.device();

        let mut states = match states {
            None => {
                let mut temp: Vec<LstmState<B, 2>> = vec![];
                for layer in self.layers.iter() {
                    temp.push(layer.init_state(batch_size, &device));
                }
                temp
            }
            _ => states.unwrap(),
        };

        let mut layer_outputs = vec![];
        for t in 0..seq_length {
            let mut input_t = x
                .clone()
                .slice([None, Some((t as i64, t as i64 + 1)), None])
                .squeeze::<2>(1);
            for (i, lstm_cell) in self.layers.iter().enumerate() {
                let mut state: LstmState<B, 2> =
                    LstmState::new(states[i].cell.clone(), states[i].hidden.clone());
                state = lstm_cell.forward(input_t, state);
                input_t = state.hidden.clone();
                states[i] = state;
            }
            layer_outputs.push(input_t);
        }

        // Stack output along sequence dimension
        let output = Tensor::stack(layer_outputs, 1);

        (output, states)
    }
}

/// Complete LSTM network with bidirectional support.
///
/// In bidirectional mode:
/// - Forward LSTM processes sequence from left to right
/// - Backward LSTM processes sequence from right to left
/// - Outputs are concatenated for final prediction
#[derive(Module, Debug)]
pub struct LstmNetwork<B: Backend> {
    // Forward direction LSTM
    pub stacked_lstm: StackedLstm<B>,
    // Optional backward direction LSTM for bidirectional processing
    pub reverse_lstm: Option<StackedLstm<B>>,
    pub dropout: Dropout,
    pub fc: Linear<B>,
}

#[derive(Config, Debug)]
pub struct LstmNetworkConfig {
    #[config(default = 1)]
    pub input_size: usize, // Single feature (number sequence)
    #[config(default = 32)]
    pub hidden_size: usize, // Size of LSTM hidden state
    #[config(default = 2)]
    pub num_layers: usize, // Number of LSTM layers
    #[config(default = 1)]
    pub output_size: usize, // Predict one number
    #[config(default = 0.1)]
    pub dropout: f64,
    #[config(default = true)]
    pub bidirectional: bool, // Use bidirectional LSTM
}

impl LstmNetworkConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LstmNetwork<B> {
        // Forward direction LSTM
        let stacked_lstm = StackedLstmConfig::new(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.dropout,
        )
        .init(device);

        // Optional backward direction LSTM for bidirectional processing
        let (reverse_lstm, hidden_size) = if self.bidirectional {
            let lstm = StackedLstmConfig::new(
                self.input_size,
                self.hidden_size,
                self.num_layers,
                self.dropout,
            )
            .init(device);
            (Some(lstm), 2 * self.hidden_size)
        } else {
            (None, self.hidden_size)
        };

        let fc = LinearConfig::new(hidden_size, self.output_size).init(device);
        let dropout = DropoutConfig::new(self.dropout).init();

        LstmNetwork {
            stacked_lstm,
            reverse_lstm,
            dropout,
            fc,
        }
    }
}

impl<B: Backend> LstmNetwork<B> {
    /// Forward pass of the network.
    ///
    /// For bidirectional processing:
    /// 1. Process sequence normally with forward LSTM
    /// 2. Process reversed sequence with backward LSTM
    /// 3. Concatenate both outputs
    /// 4. Apply final linear transformation
    ///
    /// Args:
    ///     x: Input tensor of shape (batch_size, seq_length, input_size)
    ///     states: Optional initial states
    ///
    /// Returns:
    ///     Output tensor of shape (batch_size, output_size)
    pub fn forward(&self, x: Tensor<B, 3>, states: Option<Vec<LstmState<B, 2>>>) -> Tensor<B, 2> {
        let seq_length = x.dims()[1] as i64;
        // Forward direction
        let (mut output, _states) = self.stacked_lstm.forward(x.clone(), states);

        output = match &self.reverse_lstm {
            Some(reverse_lstm) => {
                //Process sequence in reverse direction
                let (mut reverse_output, _states) = reverse_lstm.forward(x.flip([1]), None);
                // Flip back to align with forward sequence
                reverse_output = reverse_output.flip([1]);
                // Concatenate forward and backward outputs along the feature dimension
                output = Tensor::cat(vec![output, reverse_output], 2);
                output
            }
            None => output,
        };

        // Apply dropout before final layer
        output = self.dropout.forward(output);
        // Use final timestep output for prediction
        self.fc.forward(
            output
                .slice([None, Some((seq_length - 1, seq_length)), None])
                .squeeze::<2>(1),
        )
    }
}
