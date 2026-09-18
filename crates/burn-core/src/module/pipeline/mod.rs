//! Pipeline parallelism: a model split by whole layers across devices, with the activations moved
//! between them.

mod base;
mod stage_map;

pub use base::*;
pub use stage_map::*;

#[cfg(test)]
mod tests;
