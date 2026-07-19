//! A small, self-contained dependency-graph toolkit used by the fusion search.
//!
//! Everything here is generic over the [GraphNode] trait — nothing depends on the tensor IR — so
//! the algorithms can be unit-tested in isolation and reused at different granularities: fusion
//! blocks, execution-strategy chunks, or single operations.

mod dag;
mod lifetime;
mod node;
mod subgraph;

pub use dag::*;
pub use lifetime::*;
pub use node::*;
pub use subgraph::*;

#[cfg(test)]
mod tests;
