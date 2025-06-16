use burn_ir::HandleContainer;

use crate::FusionRuntime;

/// The mode in which the execution is done.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ExecutionMode {
    Lazy,
    Sync,
}

/// General trait to abstract how a single operation is executed.
pub trait Operation<R: FusionRuntime>: Send + Sync + core::fmt::Debug {
    /// Execute the operation.
    fn execute(&self, handles: &mut HandleContainer<R::FusionHandle>);
}
