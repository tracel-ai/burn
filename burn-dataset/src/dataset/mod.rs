mod dataset;
#[cfg(feature = "fake")]
mod fake;
mod in_memory;
mod iterator;

#[cfg(feature = "fake")]
pub use self::fake::*;
pub use dataset::*;
pub use in_memory::*;
pub use iterator::*;
