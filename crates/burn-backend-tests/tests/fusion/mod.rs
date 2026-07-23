pub use super::*;

mod cat;
mod fusion_f16_broadcast;
mod fusion_f16_write_vectorization;
mod fusion_shape;
// Asserts on launch-level decisions from `burn-cubecl-fusion`, which only the `cube`
// feature pulls in; the rest of this suite also runs on non-cube fusion backends (flex).
#[cfg(feature = "cube")]
mod inplace;
mod int_bitwise;
mod reduce_broadcast_vectorization;
mod reduce_broadcasted;
mod reduce_logical;

use burn_tensor::StreamId;
use std::sync::atomic::{AtomicU64, Ordering};

/// Returns a unique `StreamId` for test isolation.
pub fn test_stream() -> StreamId {
    static COUNTER: AtomicU64 = AtomicU64::new(1000);
    StreamId {
        value: COUNTER.fetch_add(1, Ordering::Relaxed),
    }
}
