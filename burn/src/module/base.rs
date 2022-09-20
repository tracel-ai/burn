use super::{State, StateNamed};
use crate::optim::Optimizer;
use crate::tensor::{
    backend::{ADBackend, Backend},
    Gradients,
};
pub use burn_derive::Module;

#[derive(Debug, new)]
pub struct LoadingError {
    message: String,
}

impl std::fmt::Display for LoadingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("Loading error: {}", self.message).as_str())
    }
}

impl std::error::Error for LoadingError {}

/// Trait for all neural network modules.
///
/// Modules should be created using the [derive](burn_derive::Module) attribute.
/// This will make your module trainable and serialization via (state)[Variable::state],
/// (load)[Variable::load] and (update_params)[Variable::update_params].
///
/// Module concret types should define their parameters via the (Param)[crate::module::Param]
/// struct.
///
/// # Example
///
/// A module should have a (backend)[crate::tensor::backend::Backend] defined as a generic
/// parameter B. This will be used by the (derive)[burn_derive::Module] attribute to generate the code
/// necessary to optimize and train the module on any backend.
///
/// To define to forward pass of your module, you should implement (Forward)[Forward].
///
/// ```rust
/// use burn::nn;
/// use burn::module::{Param, Module};
/// use burn::module::Forward;
/// use burn::tensor::Tensor;
/// use burn::tensor::backend::Backend;
///
/// #[derive(Module, Debug)]
/// struct MyModule<B: Backend> {
///   my_param: Param<nn::Linear<B>>,
///   repeat: usize,
/// }
///
/// impl<B: Backend> Forward<Tensor<B, 2>, Tensor<B, 2>> for MyModule<B> {
///    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
///        let mut x = input;
///
///        for _ in 0..self.repeat {
///            x = self.my_param.forward(x);
///        }
///
///        x
///    }
/// }
/// ```
pub trait Module:
    Send + Sync + std::fmt::Debug + std::fmt::Display + Variable<Self::Backend>
{
    type Backend: Backend;

    /// Get the name of the module.
    fn name(&self) -> &str;
}

/// Trait that defines utilities for [parameters](crate::module::Param) and [modules](Module).
///
/// The trait is automatically implemented for [modules](Module) when using de [derive](burn_derive::Module).
pub trait Variable<B: Backend> {
    /// Get the device list of the variable and all of its children.
    fn devices(&self) -> Vec<B::Device>;
    /// Move the variable and all of its children to the given device.
    fn to_device(&mut self, device: B::Device);
    /// Load the state into the variable.
    fn load(&mut self, state: &State<B::Elem>) -> Result<(), LoadingError>;
    /// Get the state of the variable.
    fn state(&self) -> State<B::Elem>;
    /// Get the number of parameters the variable has, including all of its children.
    fn num_params(&self) -> usize;
    /// Update the parameters of the variable with the given [gradients](Gradients) and [optimizer](Optimizer).
    fn update_params<O: Optimizer<Backend = B>>(&mut self, grads: &Gradients, optim: &mut O)
    where
        B: ADBackend;
    /// Load the [optimizer](Optimizer) state for the variable, including all of its children.
    ///
    /// # Note
    ///
    /// This method should only be called by generated code, see [load](Optimizer::load) to load
    /// the state of the optimizer.
    fn load_optim_state<O: Optimizer<Backend = B>>(
        &self,
        optim: &mut O,
        state_optim: &StateNamed<B::Elem>,
    ) where
        B: ADBackend;
    /// Register the [optimizer](Optimizer) state for the variable, including all of its children.
    ///
    /// # Note
    ///
    /// This method should only be called by generated code, see [state](Optimizer::state) to get
    /// the state of the optimizer.
    fn register_optim_state<O: Optimizer<Backend = B>>(
        &self,
        optim: &O,
        state_optim: &mut StateNamed<B::Elem>,
    ) where
        B: ADBackend;
}

/// Module with auto-differentiation backend.
pub trait ADModule: Module + Send + Sync + std::fmt::Debug + std::fmt::Display {
    type ADBackend: ADBackend;
    type InnerModule: Module<Backend = <Self::ADBackend as ADBackend>::InnerBackend>;

    /// Get the same module, but on the inner backend without auto-differentiation.
    fn inner(&self) -> Self::InnerModule;
}

/// Trait that can be implemented by (module)[Module] implementations to define the forward pass.
///
/// # Note
///
/// A module can implement multiple times this trait to support different input and output types.
pub trait Forward<In, Out> {
    /// The forward method mapping the input to the output.
    fn forward(&self, input: In) -> Out;
}
