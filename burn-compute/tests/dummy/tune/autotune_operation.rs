#[derive(new)]
/// Extended kernel that accounts for additional parameters, i.e. needed
/// information that does not count as an input/output.
pub struct AutotuneOperation<S: ComputeServer> {
    kernel: S::Kernel,
    parameters: Option<Vec<Handle<S>>>,
}

impl<S: ComputeServer> AutotuneOperation<S> {
    // TODO change to trait with execute. And rename to AutotuneOperation
    /// Executes the operation on given handles and server, with the additional parameters
    pub fn execute(&self, inputs: &[&Handle<S>], server: &mut S) {
        let mut handles = inputs.to_vec();

        let parameters = match self.parameters.clone() {
            Some(parameter_handles) => parameter_handles.into_iter().collect::<Vec<Handle<S>>>(),
            None => Vec::new(),
        };
        handles.extend(parameters.iter().collect::<Vec<&Handle<S>>>());

        server.execute(self.kernel.clone(), &handles);
    }
}

impl<S: ComputeServer> Clone for AutotuneOperation<S> {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel.clone(),
            parameters: self.parameters.clone(),
        }
    }
}