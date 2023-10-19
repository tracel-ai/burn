use crate::server::{ComputeServer, Handle};

/// Type of operation for the kernel
pub trait AutotuneOperation<S>: Send
where
    S: ComputeServer,
{
    fn key(&self) -> String {
        let mut key = String::new();
        key.push_str(&self.operation_key());
        key.push_str(&self.input_key());
        key
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
    pub fn execute(&self, inputs: Vec<Handle<S>>, server: &mut S) {
        let mut all_handles = inputs;
        if let Some(vec) = self.parameters.clone() {
            all_handles.extend(vec);
        }
        let slice = &all_handles
            .iter()
            .map(|h| h as &Handle<S>)
            .collect::<Vec<&Handle<S>>>();
        server.execute(self.kernel.clone(), slice);
    }

    pub fn get_kernel(self) -> S::Kernel {
        self.kernel
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
