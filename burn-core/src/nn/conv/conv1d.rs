use crate as burn;

use crate::{
    config::Config,
    module::{Module, Param},
    tensor::{backend::Backend, Tensor},
};

/// Configuration to create a one dimensional [convolution](Conv1d) layer.
#[derive(Config)]
pub struct Conv1dConfig {
    /// The number of input channels.
    pub channels_input: usize,
    /// The number of output channels.
    pub channels_output: usize,
    /// The kernel size.
    pub kernel_size: usize,
    /// The stride.
    #[config(default = 1)]
    pub stride: usize,
    /// Padding config
    pub padding: Option<Conv1dPaddingConfig>,
    /// Dilatation.
    #[config(default = 1)]
    pub dilatation: usize,
    /// The nuber of groups.
    #[config(default = 1)]
    pub groups: usize,
    /// If a bias should be applied during the convolution.
    #[config(default = true)]
    pub bias: bool,
}

/// Padding config.
#[derive(Config)]
pub enum Conv1dPaddingConfig {
    Same,
    Zeros(usize),
}

/// Conv1d layer.
#[derive(Module, Debug)]
pub struct Conv1d<B: Backend> {
    weight: Param<Tensor<B, 3>>,
    bias: Param<Option<Tensor<B, 2>>>,
}
