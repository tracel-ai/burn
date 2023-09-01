/// Weight decay module for optimizers.
pub mod decay;

/// Momentum module for optimizers.
pub mod momentum;

mod adagrad;
mod adam;
mod adamw;
mod base;
mod grad_accum;
mod grads;
mod rmsprop;
mod sgd;
mod simple;
mod visitor;

pub use adagrad::*;
pub use adam::*;
pub use adamw::*;
pub use base::*;
pub use grad_accum::*;
pub use grads::*;
pub use rmsprop::*;
pub use sgd::*;
pub use simple::*;
