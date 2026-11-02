//! Tensors split over a group of devices, the way tensor parallelism splits a model's layers.
//!
//! Each tensor has a [`Sharding`]: replicated on every member, split along one dim, or a
//! partial sum. An op with a rule runs on every member's shards at once, the rule saying how its
//! inputs must be sharded and how its output comes out; any other op gathers its inputs and runs
//! whole on every member. [`GroupBackend`] runs any backend's ops this way, one interpreter per member.

mod backend;
mod resharding;
mod rules;
mod sharding;

pub use backend::{GroupBackend, GroupDevice, GroupError, MemberDevice};
pub use sharding::{Sharding, ShardingError};

use resharding::Resharding;
use rules::{ElementwiseRule, MatmulRule, ReducedDims, ReshapeRule, SumRule};
use sharding::{GroupMember, OpSharding};
