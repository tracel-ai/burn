//! Pipeline parallelism: a model split by whole layers across devices.

mod base;
mod layout;
mod placed;
mod placement;

pub use base::*;
pub use layout::*;
pub use placed::*;
pub use placement::*;
