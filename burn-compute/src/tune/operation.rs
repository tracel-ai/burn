use crate::server::{ComputeServer, Handle};

/// Type of operation for the kernel
pub trait AutotuneOperation<S>: Send
where
    S: ComputeServer,
{
    fn key(&self) -> String {
        format!(
            "(AutoTuneKey) Operation: {} - Inputs: {:?}",
            self.operation_key(),
            self.input_key()
        )
    }
    fn operation_key(&self) -> String;
    fn input_key(&self) -> String;
    fn autotunables(&self) -> Vec<Operation<S>>;
    fn inputs(&self) -> Vec<Vec<u8>>;
    fn fastest(&self, fastest_index: usize) -> Operation<S>;
}

#[derive(new)]
pub struct Operation<S: ComputeServer> {
    kernel: S::Kernel,
    parameters: Option<Vec<Handle<S>>>,
}

impl<S: ComputeServer> Operation<S> {
    pub fn execute(&self, inputs: &[&Handle<S>], server: &mut S) {
        let mut handles = inputs.iter().cloned().collect::<Vec<_>>();

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
