/// Constant learning rate scheduler
pub mod constant;

/// Linear learning rate scheduler
pub mod linear;

/// Noam learning rate scheduler
pub mod noam;

mod base;

pub use base::*;
