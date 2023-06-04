use crate as burn;
use crate::nn::LinearRecord;

use crate::module::Module;
use crate::nn::Initializer;
use crate::nn::Linear;
use crate::nn::LinearConfig;
use burn_tensor::backend::Backend;
use burn_tensor::Tensor;

/// A GateController represents a gate in an LSTM cell. An
/// LSTM cell generally contains three gates: an input gate,
/// forget gate, and cell gate.
///
/// An Lstm gate is modeled as two linear transformations.
/// The results of these transformations are used to calculate
/// the gate's output.
#[derive(Module, Debug)]
pub struct GateController<B: Backend> {
    /// Represents the affine transformation applied to input vector
    input_transform: Linear<B>,
    /// Represents the affine transformation applied to the hidden state
    hidden_transform: Linear<B>,
}

impl<B: Backend> GateController<B> {
    pub fn new(d_input: usize, d_output: usize, bias: bool, initializer: Initializer) -> Self {
        Self {
            input_transform: LinearConfig {
                d_input,
                d_output,
                bias,
                initializer: initializer.clone(),
            }
            .init(),
            hidden_transform: LinearConfig {
                d_input: d_output,
                d_output,
                bias,
                initializer,
            }
            .init(),
        }
    }

    pub fn new_with(linear_config: &LinearConfig, record: GateControllerRecord<B>) -> Self {
        let l1 = LinearConfig::init_with(linear_config, record.input_transform);
        let l2 = LinearConfig::init_with(linear_config, record.hidden_transform);

        Self {
            input_transform: l1,
            hidden_transform: l2,
        }
    }

    pub fn get_input_weight(&self) -> Tensor<B, 2> {
        self.input_transform.get_weight()
    }

    pub fn get_hidden_weight(&self) -> Tensor<B, 2> {
        self.hidden_transform.get_weight()
    }

    pub fn get_input_bias(&self) -> Option<Tensor<B, 1>> {
        self.input_transform.get_bias()
    }

    pub fn get_hidden_bias(&self) -> Option<Tensor<B, 1>> {
        self.hidden_transform.get_bias()
    }

    #[cfg(test)]
    /// Used to initialize a gate controller with known weight layers,
    /// allowing for predictable behavior. Used only for testing in
    /// lstm.
    pub fn create_with_weights(
        d_input: usize,
        d_output: usize,
        bias: bool,
        initializer: Initializer,
        input_record: LinearRecord<B>,
        hidden_record: LinearRecord<B>,
    ) -> Self {
        let l1 = LinearConfig {
            d_input,
            d_output,
            bias,
            initializer: initializer.clone(),
        }
        .init_with(input_record);
        let l2 = LinearConfig {
            d_input,
            d_output,
            bias,
            initializer,
        }
        .init_with(hidden_record);
        Self {
            input_transform: l1,
            hidden_transform: l2,
        }
    }
}
