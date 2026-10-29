//! Layer parallelism: a model split by whole layers across devices.

mod base;
mod layer;
mod model;
#[cfg(feature = "std")]
mod pipeline;
mod placement;

pub use base::*;
pub use layer::*;
pub use model::*;
#[cfg(feature = "std")]
pub use pipeline::*;
pub use placement::*;
