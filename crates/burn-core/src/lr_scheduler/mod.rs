/// Constant learning rate scheduler
pub mod constant;

/// Linear learning rate scheduler
pub mod linear;

/// Noam learning rate scheduler
pub mod noam;

/// Exponential learning rate scheduler
pub mod exponential;

/// Cosine learning rate scheduler
pub mod cosine;

mod base;

pub use base::*;
