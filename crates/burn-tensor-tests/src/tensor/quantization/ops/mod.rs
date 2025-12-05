pub use super::*;

mod matmul;

// TODO: re-enable for cubecl backends when inputs are valid for packed U32 storage
#[cfg(feature = "ndarray")]
mod extended;
