use crate as burn;

use burn_tensor::activation;

use crate::config::Config;
use crate::module::Module;
use crate::nn::lstm::gate_controller;
use crate::nn::Initializer;
use crate::nn::LinearConfig;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

use super::gate_controller::GateController;

#[derive(Config)]
pub struct GruConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If a bias should be applied during the Gru transformation.
    pub bias: bool,
    /// Gru initializer
    /// TODO: Make default Xavier initialization. https://github.com/burn-rs/burn/issues/371
    #[config(default = "Initializer::Uniform(0.0, 1.0)")]
    pub initializer: Initializer,
    /// The batch size.
    pub batch_size: usize,
}

/// The Gru module. This implementation is for a unidirectional, stateless, Gru.
#[derive(Module, Debug)]
pub struct Gru<B: Backend> {
    update_gate: GateController<B>,
    reset_gate: GateController<B>,
    new_gate: GateController<B>,
    batch_size: usize,
    d_hidden: usize,
}

impl GruConfig {
    /// Initialize a new [Gru](Gru) module.
    pub fn init<B: Backend>(&self) -> Gru<B> {
        let d_output = self.d_hidden;

        let update_gate = gate_controller::GateController::new(
            self.d_input,
            d_output,
            self.bias,
            self.initializer.clone(),
        );
        let reset_gate = gate_controller::GateController::new(
            self.d_input,
            d_output,
            self.bias,
            self.initializer.clone(),
        );
        let new_gate = gate_controller::GateController::new(
            self.d_input,
            d_output,
            self.bias,
            self.initializer.clone(),
        );

        Gru {
            update_gate,
            reset_gate,
            new_gate,
            batch_size: self.batch_size,
            d_hidden: self.d_hidden,
        }
    }

    /// Initialize a new [gru](Gru) module.
    pub fn init_with<B: Backend>(self, record: GruRecord<B>) -> Gru<B> {
        let linear_config = LinearConfig {
            d_input: self.d_input,
            d_output: self.d_hidden,
            bias: self.bias,
            initializer: self.initializer.clone(),
        };

        Gru {
            update_gate: gate_controller::GateController::new_with(
                &linear_config,
                record.update_gate,
            ),
            reset_gate: gate_controller::GateController::new_with(
                &linear_config,
                record.reset_gate,
            ),
            new_gate: gate_controller::GateController::new_with(&linear_config, record.new_gate),
            batch_size: self.batch_size,
            d_hidden: self.d_hidden,
        }
    }
}

impl<B: Backend> Gru<B> {
    pub fn forward(
        &mut self,
        batched_input: Tensor<B, 3>,
        state: Option<Tensor<B, 2>>,
    ) -> (Tensor<B, 3>) {
        todo!()
    }
}
