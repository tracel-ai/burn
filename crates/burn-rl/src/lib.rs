#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! A library for training reinforcement learning agents.

/// Module for implementing an agent.
pub mod agent;
/// Module for implementing an environment.
pub mod environment;
/// Transition buffer.
pub mod transition_buffer;

pub use agent::*;
pub use environment::*;
pub use transition_buffer::*;

#[cfg(test)]
pub(crate) type TestBackend = burn_ndarray::NdArray<f32>;

#[cfg(test)]
pub(crate) mod tests {}
