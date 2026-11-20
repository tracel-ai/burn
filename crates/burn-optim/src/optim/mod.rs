#[macro_use]
mod state_macros;
/// Weight decay module for optimizers.
pub mod decay;

/// Momentum module for optimizers.
pub mod momentum;

mod adafactor;
mod adagrad;
mod adam;
mod adamw;
mod adan;
mod base;
mod configurable_optimizer;
mod grad_accum;
mod grads;
mod lamb;
mod lbfgs;
mod lion;
mod module;
mod muon;
mod precision;
mod rmsprop;
mod sgd;
mod state;
mod visitor;

pub use adafactor::*;
pub use adagrad::*;
pub use adam::*;
pub use adamw::*;
pub use adan::*;
pub use base::*;
pub use burn_derive::RecordState;
pub use configurable_optimizer::*;
pub use grad_accum::*;
pub use grads::*;
pub use lamb::*;
pub use lbfgs::*;
pub use lion::*;
pub use module::*;
pub use muon::*;
pub use precision::*;
pub use rmsprop::*;
pub use sgd::*;
pub use state::*;

#[cfg(test)]
mod test_utils;
