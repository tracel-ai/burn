use super::{State, StateNamed};
use crate::optim::Optimizer;
use crate::tensor::backend::{ADBackend, Backend};
pub use burn_derive::Module;

/// Trait for all neural network modules.
///
/// Modules should be created using the [derive](burn_derive::Module) attribute.
/// This will make your module trainable, savable and loadable via [update_params](Module::update_params),
/// [state](Module::state) and [load](Module::load).
///
/// Module concrete types should define their parameters via the [Param](crate::module::Param)
/// struct.
///
/// # Example
///
/// A module should have a [backend](crate::tensor::backend::Backend) defined as a generic
/// parameter B. This will be used by the [derive](burn_derive::Module) attribute to generate the code
/// necessary to optimize and train the module on any backend.
///
/// ```rust
/// use burn::nn;
/// use burn::module::{Param, Module};
/// use burn::tensor::Tensor;
/// use burn::tensor::backend::Backend;
///
/// #[derive(Module, Debug)]
/// struct MyModule<B: Backend> {
///   my_param: Param<nn::Linear<B>>,
///   my_other_field: usize,
/// }
/// ```
pub trait Module: Send + Sync + std::fmt::Debug + std::fmt::Display {
    type Backend: Backend;

    /// Get the device list of the module and all of its sub-modules.
    fn devices(&self) -> Vec<<Self::Backend as Backend>::Device>;
    /// Move the module and all of its sub-modules to the given device.
    fn to_device(&mut self, device: <Self::Backend as Backend>::Device);
    /// Load the module state.
    fn load(&mut self, state: &State<<Self::Backend as Backend>::Elem>)
        -> Result<(), LoadingError>;
    /// Get the module state.
    fn state(&self) -> State<<Self::Backend as Backend>::Elem>;
    /// Detach the module from the graph.
    fn detach(&mut self);
    /// Get the number of parameters the module has, including all of its sub-modules.
    fn num_params(&self) -> usize;
    /// Update the module parameters with the given gradients and [optimizer](Optimizer).
    fn update_params<O: Optimizer<Backend = Self::Backend>>(
        &mut self,
        grads: &<Self::Backend as ADBackend>::Gradients,
        optim: &mut O,
    ) where
        Self::Backend: ADBackend;
    /// Load the [optimizer](Optimizer) state for the module, including all of its sub-modules.
    ///
    /// # Note
    ///
    /// This method should only be called by generated code, see [load](Optimizer::load) to load
    /// the state of the optimizer.
    fn load_optim_state<O: Optimizer<Backend = Self::Backend>>(
        &self,
        optim: &mut O,
        state_optim: &StateNamed<<Self::Backend as Backend>::Elem>,
    ) where
        Self::Backend: ADBackend;
    /// Register the [optimizer](Optimizer) state for the module, including all of its sub-modules.
    ///
    /// # Note
    ///
    /// This method should only be called by generated code, see [state](Optimizer::state) to get
    /// the state of the optimizer.
    fn register_optim_state<O: Optimizer<Backend = Self::Backend>>(
        &self,
        optim: &O,
        state_optim: &mut StateNamed<<Self::Backend as Backend>::Elem>,
    ) where
        Self::Backend: ADBackend;
}

/// Module with auto-differentiation backend.
pub trait ADModule:
    Module<Backend = Self::ADBackend> + Send + Sync + std::fmt::Debug + std::fmt::Display
{
    type ADBackend: ADBackend;
    type InnerModule: Module<Backend = <Self::ADBackend as ADBackend>::InnerBackend>;

    /// Get the same module, but on the inner backend without auto-differentiation.
    fn inner(&self) -> Self::InnerModule;
}

#[derive(new, Debug)]
pub struct LoadingError {
    message: String,
}

impl std::fmt::Display for LoadingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("Loading error: {}", self.message).as_str())
    }
}

impl std::error::Error for LoadingError {}
