use burn_ir::{HandleContainer, OperationIr, TensorError};

/// The failure claiming any tensor `op` reads — the check a unit of work
/// makes before it runs.
///
/// Work whose input was never written must not run: those bytes are whatever
/// the allocation happened to hold, and computing on them turns a failure
/// that named one tensor into a wrong answer that names none. The outputs
/// take the same claim instead, so a read below the skip still reports the
/// failure that started it.
///
/// `inputs()` rather than `nodes()`: this runs before every operation on the
/// hot path, and `nodes()` collects into a fresh `Vec` to chain the two.
pub(crate) fn input_failure<'a, H>(
    op: &OperationIr,
    handles: &'a HandleContainer<H>,
) -> Option<&'a TensorError>
where
    H: Clone,
{
    // Nothing is claimed, so nothing can be found — and asking anyway would
    // cost a boxed iterator per operation for an answer that is always
    // `None`. This runs before every operation, so the check has to be free
    // while nothing has failed.
    if !handles.has_claims() {
        return None;
    }

    // A drop names its tensor as an input, but it does not read it — it is
    // what releases it, and releasing is how a claim stops being held. Skip
    // it and the claim outlives every tensor that could report it, for the
    // life of the server: the bound this whole design rests on is that a
    // claim lives exactly as long as the tensor carrying it.
    if let OperationIr::Drop(_) = op {
        return None;
    }

    op.inputs().find_map(|node| handles.error(&node.id))
}

/// Claim every tensor `op` was going to write, so a read of one reports
/// `error` instead of handing back bytes nothing wrote.
///
/// Unconditional, because these are exactly the tensors this operation was
/// responsible for: an in-place output has a handle registered while the
/// launch is still being planned, so finding one there says nothing about
/// whether the kernel that fills it ever ran.
pub(crate) fn claim_outputs<H>(
    op: &OperationIr,
    handles: &mut HandleContainer<H>,
    error: &TensorError,
) where
    H: Clone,
{
    for node in op.outputs() {
        handles.claim(node.id, error.clone());
    }
}

/// The message inside a caught panic payload. Covers what `panic!` produces:
/// `&'static str` and `String`.
pub(crate) fn panic_message(panic: &(dyn core::any::Any + Send)) -> &str {
    panic
        .downcast_ref::<&'static str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic payload>")
}
