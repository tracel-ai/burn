//! Tensors split over a group of devices, the way tensor parallelism splits a model's layers.
//!
//! Each tensor has a [`GroupPlacement`]: replicated on every rank, split along one dim, or a
//! partial sum. Each op family has a rule that says where its inputs must be for the op to run on
//! every rank at once, and where its output lands. [`GroupBackend`] runs any backend's ops this
//! way, one interpreter per rank.

mod backend;
mod placement;
mod redistribution;
mod rules;

pub use backend::{GroupBackend, GroupClient, GroupDevice};
pub use placement::GroupPlacement;

use placement::{Chunks, OpPlacement};
use redistribution::Redistribution;
use rules::*;
