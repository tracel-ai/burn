use burn_core::tensor::{Device, is_capturing};

/// Run one tensor operation on its own, outside any fused block, while a graph is captured.
///
/// A captured graph (see `burn::tensor::capture`) replays against the buffers it recorded, so an
/// optimizer state (a moment, a velocity, the parameter itself) only advances across replays when
/// its update is written where it was read. The tensor API promises nothing about buffers; what
/// the backends do is reuse an input no one else holds:
///
/// - Without fusion, an op reuses its left operand.
/// - With fusion, a fused block reuses the first input it reads, which depends on every op fused
///   around it.
///
/// Flushing right before and after `op` makes it a block of its own, so it reuses its left
/// operand either way. For that, `op` must be a **single** tensor operation whose operands are
/// already computed, with the state as its left operand and no other owner: an update made of
/// several ops is several `isolated` calls, each landing in the buffer of the previous one.
///
/// Outside a capture, `op` runs as is: where an update lands doesn't matter there, and the
/// flushes would keep fusion from fusing the optimizer step.
pub(crate) fn isolated<T>(device: &Device, op: impl FnOnce() -> T) -> T {
    if !is_capturing() {
        return op();
    }

    device.flush();
    let output = op();
    device.flush();
    output
}
