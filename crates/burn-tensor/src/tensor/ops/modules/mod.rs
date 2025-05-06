/// Module with convolution operations.
pub mod conv;

/// Module with cat operation
pub(crate) mod cat;
/// Module with repeat operation
pub(crate) mod repeat_dim;
/// Module with unfold operations.
pub(crate) mod unfold;

/// Module with pooling operations.
pub mod pool;

mod base;

/// Module with linear operations.
pub mod linear;

pub use base::*;
