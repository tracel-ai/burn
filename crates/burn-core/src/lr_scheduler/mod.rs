/// Constant learning rate scheduler
pub mod constant;

/// Linear learning rate scheduler
pub mod linear;

/// Noam learning rate scheduler
pub mod noam;

/// Exponential learning rate scheduler
pub mod exponential;

mod base;

pub use base::*;
