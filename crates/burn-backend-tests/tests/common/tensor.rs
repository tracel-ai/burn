// Burn backend tensor tests, reusable with element types.

pub use super::*;

#[path = "../tensor/clone_invariance.rs"]
mod clone_invariance;

#[cfg(feature = "std")]
#[path = "../tensor/multi_threads.rs"]
mod multi_threads;

#[cfg(feature = "distributed")]
#[path = "../tensor/distributed.rs"]
mod distributed;

// Default float dtype
#[path = "../tensor/float/mod.rs"]
mod float;

// Default integer dtype
#[path = "../tensor/int/mod.rs"]
mod int;

// Default bool dtype
#[path = "../tensor/bool/mod.rs"]
mod bool;
