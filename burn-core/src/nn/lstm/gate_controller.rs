use crate as burn;
use crate::nn::LinearRecord;

use crate::module::Module;
use crate::nn::Initializer;
use crate::nn::Linear;
use crate::nn::LinearConfig;
use burn_tensor::backend::Backend;
use burn_tensor::Tensor;

#[derive(Module, Debug)]
pub struct GateController<B: Backend> {
    /// Represents the affine transformation applied to input vector
    input_transform: Linear<B>,
    /// Represents the affine transformation applied to the hidden state
    hidden_transform: Linear<B>,
}

impl<B: Backend> GateController<B> {
    pub fn new(
        d_input: usize,
        d_output: usize,
        bias: bool,
        initializer: Initializer,
    ) -> GateController<B> {
        GateController {
            input_transform: LinearConfig {
                d_input: d_input,
                d_output: d_output,
                bias: bias,
                initializer: initializer.clone(),
            }
            .init(),
            hidden_transform: LinearConfig {
                d_input: d_output,
                d_output: d_output,
                bias: bias,
                initializer: initializer.clone(),
            }
            .init(),
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

    /// Only used for testing in lstm
    pub fn create_with_weights(
        d_input: usize,
        d_output: usize,
        bias: bool,
        initializer: Initializer,
        input_record: LinearRecord<B>,
        hidden_record: LinearRecord<B>,
    ) -> GateController<B> {
        let l1 = LinearConfig {
            d_input: d_input,
            d_output: d_output,
            bias: bias,
            initializer: initializer.clone(),
        }
        .init_with(input_record);
        let l2 = LinearConfig {
            d_input: d_input,
            d_output: d_output,
            bias: bias,
            initializer: initializer.clone(),
        }
        .init_with(hidden_record);
        GateController {
            input_transform: l1,
            hidden_transform: l2,
        }
    }
}
