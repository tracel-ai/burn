/// Constant learning rate scheduler
pub mod constant;

/// Composed learning rate scheduler
pub mod composed;

/// Linear learning rate scheduler
pub mod linear;

/// Noam learning rate scheduler
pub mod noam;

/// Exponential learning rate scheduler
pub mod exponential;

/// Cosine learning rate scheduler
pub mod cosine;

/// Step learning rate scheduler
pub mod step;

mod base;

pub use base::*;
