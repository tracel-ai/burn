use burn_core as burn;

use crate::{Linear, LinearConfig, LinearLayout};
use burn::module::{Initializer, Module};
use burn::tensor::{Device, Tensor};

/// A GateController represents a gate in an LSTM cell. An
/// LSTM cell generally contains three gates: an input gate,
/// forget gate, and output gate. Additionally, cell gate
/// is just used to compute the cell state.
///
/// An Lstm gate is modeled as two linear transformations.
/// The results of these transformations are used to calculate
/// the gate's output.
#[derive(Module, Debug)]
pub struct GateController {
    /// Represents the affine transformation applied to input vector
    pub input_transform: Linear,
    /// Represents the affine transformation applied to the hidden state
    pub hidden_transform: Linear,
}

impl GateController {
    /// Initialize a new [gate_controller](GateController) module.
    pub fn new(
        d_input: usize,
        d_output: usize,
        bias: bool,
        initializer: Initializer,
        device: &Device,
    ) -> Self {
        Self {
            input_transform: LinearConfig {
                d_input,
                d_output,
                bias,
                initializer: initializer.clone(),
                layout: LinearLayout::Row,
            }
            .init(device),
            hidden_transform: LinearConfig {
                d_input: d_output,
                d_output,
                bias,
                initializer,
                layout: LinearLayout::Row,
            }
            .init(device),
        }
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
    pub fn gate_product(&self, input: Tensor<2>, hidden: Tensor<2>) -> Tensor<2> {
        self.input_transform.forward(input) + self.hidden_transform.forward(hidden)
    }

    /// Used to initialize a gate controller with known weight layers,
    /// allowing for predictable behavior. Used only for testing in
    /// lstm.
    #[cfg(test)]
    pub fn create_with_weights(
        d_input: usize,
        d_output: usize,
        bias: bool,
        initializer: Initializer,
        input_record: crate::LinearRecord,
        hidden_record: crate::LinearRecord,
    ) -> Self {
        let l1 = LinearConfig {
            d_input,
            d_output,
            bias,
            initializer: initializer.clone(),
            layout: LinearLayout::Row,
        }
        .init(&input_record.weight.device())
        .load_record(input_record);
        let l2 = LinearConfig {
            d_input,
            d_output,
            bias,
            initializer,
            layout: LinearLayout::Row,
        }
        .init(&hidden_record.weight.device())
        .load_record(hidden_record);

        Self {
            input_transform: l1,
            hidden_transform: l2,
        }
    }
}
