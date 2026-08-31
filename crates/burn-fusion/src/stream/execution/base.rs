use burn_ir::HandleContainer;

use crate::FusionRuntime;
use burn_backend::ExecutionError;

/// The mode in which the execution is done.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ExecutionMode {
    Lazy,
    Sync,
}

/// General trait to abstract how a single operation is executed.
pub trait Operation<R: FusionRuntime>: Send + Sync + core::fmt::Debug {
    /// Execute the operation.
    ///
    /// Reporting a failure claims everything this operation was going to write,
    /// exactly as a panic out of it would — the difference is only that an
    /// error carries its own type and backtrace, where a panic payload is a
    /// message. Prefer returning one.
    fn execute(&self, handles: &mut HandleContainer<R::FusionHandle>)
    -> Result<(), ExecutionError>;
}
