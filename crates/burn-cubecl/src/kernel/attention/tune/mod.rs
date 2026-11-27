//! Attention's autotunes: the forward pass's routines raced per problem, at one query position or
//! many ([`AttentionAutotuneGroups`]), and the backward pass's strategies.

mod backward;
mod forward;

pub(crate) use backward::*;
pub use forward::AttentionAutotuneGroups;
pub(crate) use forward::launch_tuned;
