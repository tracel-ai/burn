mod supervised_learning;

#[cfg(feature = "ddp")]
pub(crate) mod ddp_v2;
pub(crate) mod multi_v2;
pub(crate) mod single_v2;

pub use supervised_learning::*;
