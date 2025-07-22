pub(crate) mod execution;
pub(crate) mod queue;
pub(crate) mod shared_tensors;
pub(crate) mod store;

#[cfg(feature = "memory-checks")]
/// Memory checks module.
pub mod memory_checks;

#[cfg(not(feature = "memory-checks"))]
#[macro_export]
/// Export memory checks tests.
macro_rules! memory_checks {
    () => {
        #[cfg(test)]
        mod memory_checks {
            #[ignore = "'memory-checks' disabled"]
            #[test]
            fn test_memory_leaks() {
                //
            }
        }
    };
}

mod base;
mod context;
mod multi;

pub use base::*;
pub use context::*;
pub use execution::*;
pub use multi::*;
