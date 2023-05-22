use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

use super::Initializer;

#[derive(Config)]
pub struct LSTMConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the hidden state.
    pub d_hidden: usize,
    /// If a bias should be applied during the LSTM transformation
    pub bias: bool,
    /// The type of function used to initialize LSTM parameters
    #[config(default = "Initializer::UniformDefault")]
    pub initializer: Initializer,
}