use crate as burn;
use crate::config::Config;

#[derive(Config)]
pub struct AdamConfig {
    /// Learning rate for the optimizer.
    learning_rate: f64,
    /// Parameter for Adam.
    beta_1: f64,
    /// Parameter for Adam.
    beta_2: f64,
    /// A value required for numerical stability, typically 1e-8.
    epsilon: f64,
}
