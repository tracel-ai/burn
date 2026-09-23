//! Layer parallelism: a model split by whole layers across devices.

mod base;
mod layer;
mod model;
mod placement;

pub use base::*;
pub use layer::*;
pub use model::*;
pub use placement::*;
