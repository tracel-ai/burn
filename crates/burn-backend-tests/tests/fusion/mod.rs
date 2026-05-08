pub use super::*;

mod fusion_f16_broadcast;
mod fusion_f16_write_vectorization;
mod fusion_shape;
mod reduce_broadcasted;

use burn_tensor::StreamId;
use std::sync::atomic::{AtomicU64, Ordering};

/// Returns a unique `StreamId` for test isolation.
pub fn test_stream() -> StreamId {
    static COUNTER: AtomicU64 = AtomicU64::new(1000);
    StreamId {
        value: COUNTER.fetch_add(1, Ordering::Relaxed),
    }
}
