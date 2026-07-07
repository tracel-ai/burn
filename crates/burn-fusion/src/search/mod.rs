mod block;
mod deps;
mod optimization;

pub(super) mod merging;
pub(super) use block::*;
pub(super) use deps::*;

pub use optimization::*;
