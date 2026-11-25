//! The matmul family's autotunes: **few-rows**, **many-rows** and **block-scaled**, the races cubek's
//! selection sends a measured product to ([`MatmulAutotune`](cubek::matmul::kernels::select::MatmulAutotune)).
//!
//! Every candidate is a routine of cubek's, at one of its settings. Float and packed products race
//! the same two autotunes, each candidate listed for one kind and declining the other before
//! compiling anything. Under each autotune sits a floor that serves any product: the unit routine
//! on every packed input decoded first, which asks nothing of the device.
//!
//! An autotune's groups run in the order its caller gives them ([`MatmulAutotuneGroups`]), a
//! **cutoff** rather than a tie-break: the tuner measures the highest group whose candidates
//! launch, and stops there. Inside a group a round tries the candidates in the order their cuts
//! suit the key, and the first one close enough to the device's roofline ends it before the ones
//! after it are compiled.
//!
//! A cached winner is an index into one autotune's candidates under its name, so each autotune has
//! its own tuner, named rather than taken from the module path.
//!
//! The block-scaled autotune races only the block-scaled kernel's settings, which all give the
//! same numbers.
//!
//! The indexed products have autotunes of their own, each its tuner and key: the routed product's
//! ([`routed`]) and the M-grouped product's ([`m_grouped`]), the public entries of a caller that
//! measures them.

mod base;
mod block_scaled;
mod candidate;
mod few_rows;
mod floor;
pub mod m_grouped;
mod many_rows;
mod packed_candidate;
mod roofline;
pub mod routed;
#[cfg(test)]
mod tests;

pub use base::MatmulAutotuneGroups;
#[cfg(any(feature = "fusion", test))]
pub(crate) use base::fastest_routine;
