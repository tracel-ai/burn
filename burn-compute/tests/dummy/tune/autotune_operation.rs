use burn_compute::{
    server::{ComputeServer, Handle},
    tune::AutotuneOperation,
};
use derive_new::new;

#[derive(new)]
/// Extended kernel that accounts for additional parameters, i.e. needed
/// information that does not count as an input/output.
pub struct DummyAutotuneOperation<S: ComputeServer> {
    kernel: S::Kernel,
    parameters: Option<Vec<Handle<S>>>,
}

impl<S: ComputeServer> AutotuneOperation<S> for DummyAutotuneOperation<S> {
    /// Executes the operation on given handles and server, with the additional parameters
    fn execute(&self, inputs: &[&Handle<S>], server: &mut S) {
        let mut handles = inputs.to_vec();

        let parameters = match self.parameters.clone() {
            Some(parameter_handles) => parameter_handles.into_iter().collect::<Vec<Handle<S>>>(),
            None => Vec::new(),
        };
        handles.extend(parameters.iter().collect::<Vec<&Handle<S>>>());

        server.execute(self.kernel.clone(), &handles);
    }
}