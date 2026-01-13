mod agent;
mod environment;
mod transition_buffer;

pub use agent::*;
pub use environment::*;
pub use transition_buffer::*;

#[cfg(test)]
pub(crate) type TestBackend = burn_ndarray::NdArray<f32>;

#[cfg(test)]
pub(crate) mod tests {
    use crate::TestBackend;
    use burn_core::{prelude::Tensor, tensor::Bool};
    use std::default::Default;
}
