/// Weight decay module for optimizers.
pub mod decay;

/// Momentum module for optimizers.
pub mod momentum;

mod adagrad;
mod adam;
mod adamw;
mod adamw_8bit;
// mod adamw_8bit_fused;
mod adan;
mod base;
mod grad_accum;
mod grads;
mod lbfgs;
mod muon;
mod rmsprop;
mod sgd;
mod simple;
mod visitor;

pub use adagrad::*;
pub use adam::*;
pub use adamw::*;
pub use adamw_8bit::*;
// pub use adamw_8bit_fused::*;
pub use adan::*;
pub use base::*;
pub use grad_accum::*;
pub use grads::*;
pub use lbfgs::*;
pub use muon::*;
pub use rmsprop::*;
pub use sgd::*;
pub use simple::*;
