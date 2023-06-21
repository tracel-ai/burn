/// Weight decay module for optimizers.
pub mod decay;

/// Momentum module for optimizers.
pub mod momentum;

mod adam;
mod base;
mod grad_accum;
mod grads;
mod sgd;
mod simple;
mod visitor;

pub use adam::*;
pub use base::*;
pub use grad_accum::*;
pub use grads::*;
pub use sgd::*;
pub use simple::*;
