mod block;
mod optimization;

pub(crate) mod graph;
pub(super) mod merging;
#[cfg(test)]
pub(crate) mod testing;
pub(super) use block::*;

pub use optimization::*;
