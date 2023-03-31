use alloc::{format, string::String, vec::Vec};

use super::{ParamId, State};
use crate::tensor::backend::{ADBackend, Backend};
pub use burn_derive::Module;
use burn_tensor::Tensor;

/// Trait for all neural network modules.
///
/// Modules should be created using the [derive](burn_derive::Module) attribute.
/// This will make your module trainable, savable and loadable via
/// [state](Module::state) and [load](Module::load).
///
/// # Example
///
/// A module should have a [backend](crate::tensor::backend::Backend) defined as a generic
/// parameter B. This will be used by the [derive](burn_derive::Module) attribute to generate the code
/// necessary to optimize and train the module on any backend.
///
/// ```rust
/// // Not necessary when using the burn crate directly.
/// use burn_core as burn;
///
/// use burn::{
///     nn,
///     module::Module,
///     tensor::Tensor,
///     tensor::backend::Backend,
/// };
///
/// #[derive(Module, Debug)]
/// struct MyModule<B: Backend> {
///   my_param: nn::Linear<B>,
///   my_other_field: usize,
/// }
/// ```
pub trait Module<B: Backend>: Clone + Send + Sync + core::fmt::Debug {
    /// Get the device list of the module and all of its sub-modules.
    fn devices(&self) -> Vec<B::Device>;
    /// Move the module and all of its sub-modules to the given device.
    fn to_device(self, device: &B::Device) -> Self;
    /// Load the module state.
    fn load(self, state: &State<B::FloatElem>) -> Result<Self, LoadingError>;
    /// Get the module state.
    fn state(&self) -> State<B::FloatElem>;
    /// Detach the module from the graph.
    fn detach(self) -> Self;
    /// Get the number of parameters the module has, including all of its sub-modules.
    fn num_params(&self) -> usize;
    /// Visit each tensor in the module with a [visitor](ModuleVisitor).
    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V);
    /// Map each tensor in the module with a [mapper](ModuleMapper).
    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self;
}

pub trait ModuleVisitor<B: Backend> {
    fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>);
}

pub trait ModuleMapper<B: Backend> {
    fn map<const D: usize>(&mut self, id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D>;
}

/// Module with auto-differentiation backend.
pub trait ADModule<B: ADBackend>: Module<B> + Send + Sync + core::fmt::Debug {
    type InnerModule: Module<B::InnerBackend>;

    /// Get the same module, but on the inner backend without auto-differentiation.
    fn inner(self) -> Self::InnerModule;
    fn from_inner(module: Self::InnerModule) -> Self;
}

#[derive(new, Debug)]
pub struct LoadingError {
    message: String,
}

impl core::fmt::Display for LoadingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("Loading error: {}", self.message).as_str())
    }
}

// TODO: Move from std to core after Error is core (see https://github.com/rust-lang/rust/issues/103765)
#[cfg(feature = "std")]
impl std::error::Error for LoadingError {}
