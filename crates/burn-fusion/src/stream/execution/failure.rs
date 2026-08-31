use burn_ir::{HandleContainer, OperationIr, TensorError};

/// The failure that errored any tensor `op` reads — the check a unit of work
/// makes before it runs.
///
/// Work whose input was never written must not run: those bytes are whatever
/// the allocation happened to hold, and computing on them turns a failure
/// that named one tensor into a wrong answer that names none. The outputs
/// take the same error instead, so a read below the skip still reports the
/// failure that started it.
///
/// `inputs()` rather than `nodes()`: this runs before every operation on the
/// hot path, and `nodes()` collects into a fresh `Vec` to chain the two.
pub(crate) fn input_error<'a, H>(
    op: &OperationIr,
    handles: &'a HandleContainer<H>,
) -> Option<&'a TensorError>
where
    H: Clone,
{
    // No tensor is errored, so nothing can be found — and asking anyway would
    // cost a boxed iterator per operation for an answer that is always
    // `None`. This runs before every operation, so the check has to be free
    // while nothing has failed.
    if !handles.has_errors() {
        return None;
    }

    // A drop names its tensor as an input, but it does not read it — it is
    // what releases it, and releasing is how an error stops being held. Skip
    // it and the error outlives every tensor that could report it, for the
    // life of the server: the bound this whole design rests on is that an
    // error lives exactly as long as the tensor carrying it.
    if let OperationIr::Drop(_) = op {
        return None;
    }

    op.inputs().find_map(|node| handles.error(&node.id))
}

/// Record `error` on every tensor `op` was going to write, so a read of one
/// reports it instead of handing back bytes nothing wrote.
pub(crate) fn set_output_errors<H>(
    op: &OperationIr,
    handles: &mut HandleContainer<H>,
    error: &TensorError,
) where
    H: Clone,
{
    for node in op.outputs() {
        handles.set_error(node.id, error.clone());
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
