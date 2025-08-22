use burn_common::stub::RwLock;
#[cfg(target_has_atomic = "64")]
use core::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(target_has_atomic = "64"))]
use portable_atomic::{AtomicU64, Ordering};
#[cfg(feature = "std")]
use std::thread::ThreadId;

use crate::collections::HashMap;

#[cfg(feature = "std")]
static CONTEXTS: spin::Lazy<RwLock<HashMap<ThreadId, ContextId>>> =
    spin::Lazy::new(|| RwLock::new(HashMap::new()));

/// Unique identifier generated for each context/tape.
#[derive(Clone, Hash, PartialEq, Eq, Debug, Copy)]
pub struct ContextId {
    /// The integer representation of the id
    pub value: u64,
}

impl ContextId {
    /// Create a unique [context id](ContextId).
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let value = COUNTER.fetch_add(1, Ordering::Relaxed);
        if value == u64::MAX {
            panic!("ContextId overflowed");
        }
        Self { value }
    }

    /// Get the current [context id](ContextId).
    pub fn current() -> Self {
        #[cfg(feature = "std")]
        {
            let thread_id = std::thread::current().id();

            // Fast path: try to read without writing
            if let Some(ctx) = CONTEXTS.read().unwrap().get(&thread_id) {
                return *ctx;
            }

            // Slow path: upgrade to write
            let mut map = CONTEXTS.write().unwrap();
            let context_id = ContextId::new();
            map.insert(thread_id, context_id);
            context_id
        }
        #[cfg(not(feature = "std"))]
        // Single global context
        Self { value: 0 }
    }
}

impl Default for ContextId {
    /// Create a default context
    fn default() -> Self {
        Self::current()
    }
}
