use burn_core as burn;

use burn::config::Config;

/// The reduction type for the loss.
#[derive(Config, Debug)]
pub enum Reduction {
    /// The mean of the losses will be returned.
    Mean,

    /// The sum of the losses will be returned.
    Sum,

    /// The mean of the losses will be returned.
    Auto,
}
