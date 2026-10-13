//! Tensors split over a group of devices, the way tensor parallelism splits a model's layers.
//!
//! Each tensor has a [`GroupPlacement`]: replicated on every member, split along one dim, or a
//! partial sum. An op with a rule runs on every member's shards at once, the rule saying where its
//! inputs must be and where its output lands; any other op gathers its inputs and runs whole on
//! every member. [`GroupBackend`] runs any backend's ops this way, one interpreter per member.

mod backend;
mod placement;
mod redistribution;
mod rules;

pub use backend::{GroupBackend, GroupDevice};
pub use placement::GroupPlacement;

use placement::{DimSplit, OpPlacement, PlacementShapes};
use redistribution::Redistribution;
use rules::*;
