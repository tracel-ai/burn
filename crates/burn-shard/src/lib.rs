//! Tensors split over several devices, the way tensor parallelism splits a model's layers.
//!
//! Each tensor has a [`Sharding`]: replicated on every member, split along one dim, split into
//! chunks each held by a group of members, or a partial sum. An op with a rule runs on every
//! member's shards at once, the rule saying how its inputs must be sharded and how its output
//! comes out; any other op gathers its inputs and runs whole on every member. [`ShardBackend`]
//! runs any backend's ops this way, one interpreter per member.

mod backend;
mod kind;
mod resharding;
mod rules;
mod sharding;

pub use backend::{
    CollectiveMember, CollectiveOpHandler, CollectiveOpRegistry, CustomOpPlan, CustomOpRule,
    MemberDevice, ShardBackend, ShardCustomOps, ShardError, ShardedComputeDevice,
};
pub use burn_router::CustomOpRegistry;
pub use burn_std::distributed::{Sharding, ShardingError, Shards, ShardsError};
pub use kind::{Linearity, OpKind, ReducedDims, ReductionKind};

use resharding::Resharding;
use rules::{
    AlongDimRule, AttentionRule, ElementwiseRule, LayoutRule, MatmulRule, ReduceRule, ReshapeRule,
};
use sharding::{OpInputs, OpOutputs, OpSharding, ShardMember};
