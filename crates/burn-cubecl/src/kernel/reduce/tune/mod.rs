//! The reduce family's autotunes: the routines raced per problem ([`launch_tuned`]), and the plans
//! a multi-axis sum or mean can run as ([`autotune_reduce_dims_plan`]).

mod dims_plan;
mod routines;

pub(crate) use dims_plan::*;
pub(crate) use routines::*;
