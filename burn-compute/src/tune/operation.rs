use crate::server::{ComputeServer, Handle};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Type of operation for the kernel
pub trait AutotuneOperation<S>: Send
where
    S: ComputeServer,
{
    /// The key used in the tune cache
    fn key(&self) -> String {
        format!(
            "(AutoTuneKey) Operation: {} - Inputs: {}",
            self.operation_key(),
            self.input_key()
        )
    }

    /// Partial key referring to the operation type,
    /// generally hardcoded for an autotune operation
    fn operation_key(&self) -> String;

    /// Partial key referring to the input shape
    fn input_key(&self) -> String;

    /// All candidate operations for autotuning this operation type
    fn autotunables(&self) -> Vec<Operation<S>>;

    /// Inputs generated for benchmarked executions
    fn inputs(&self) -> Vec<Vec<u8>>;

    /// Returns the operation for the given index, matching the order
    /// returned by [autotunables](AutotuneOperation::autotunable)
    fn fastest(&self, fastest_index: usize) -> Operation<S>;
}

#[derive(new)]
/// Extended kernel that accounts for additional parameters, i.e. needed
/// information that does not count as an input/output.
pub struct Operation<S: ComputeServer> {
    kernel: S::Kernel,
    parameters: Option<Vec<Handle<S>>>,
}

impl<S: ComputeServer> Operation<S> {
    /// Executes the operation on given handles and server, with the additional parameters
    pub fn execute(&self, inputs: &[&Handle<S>], server: &mut S) {
        let mut handles = inputs.to_vec();

        let p = match self.parameters.clone() {
            Some(parameter_handles) => parameter_handles.into_iter().collect::<Vec<Handle<S>>>(),
            None => Vec::new(),
        };
        handles.extend(p.iter().collect::<Vec<&Handle<S>>>());
        server.execute(self.kernel.clone(), &handles);
    }
}

impl<S: ComputeServer> Clone for Operation<S> {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
            parameters: self.parameters.clone(),
        }
    }
}
