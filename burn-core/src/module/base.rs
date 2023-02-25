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
/// // Not necessary when using the burn crate directly.
/// use burn_core as burn;
///
/// use burn::{
///     nn,
///     module::{Param, Module},
///     tensor::Tensor,
///     tensor::backend::Backend,
/// };
///
/// #[derive(Module, Debug)]
/// struct MyModule<B: Backend> {
///   my_param: Param<nn::Linear<B>>,
///   my_other_field: usize,
/// }
/// ```
pub trait Module: Send + Sync + core::fmt::Debug + core::fmt::Display {
    type Backend: Backend;

    /// Get the device list of the module and all of its sub-modules.
    fn devices(&self) -> Vec<<Self::Backend as Backend>::Device>;
    /// Move the module and all of its sub-modules to the given device.
    fn to_device(&mut self, device: &<Self::Backend as Backend>::Device);
    /// Load the module state.
    fn load(&mut self, state: &State<<Self::Backend as Backend>::Elem>)
        -> Result<(), LoadingError>;
    /// Get the module state.
    fn state(&self) -> State<<Self::Backend as Backend>::Elem>;
    /// Detach the module from the graph.
    fn detach(&mut self);
    /// Get the number of parameters the module has, including all of its sub-modules.
    fn num_params(&self) -> usize;
    /// Visit each tensor in the module with a [visitor](ModuleVisitor).
    fn visit<V: ModuleVisitor<Self::Backend>>(&self, visitor: &mut V);
    /// Visit each tensor in the module with a [visitor](ModuleVisitorMut).
    ///
    /// Note that each tensor is mutable and may be updated by the visitor.
    fn visit_mut<V: ModuleVisitorMut<Self::Backend>>(&mut self, visitor: &mut V);
}

pub trait ModuleVisitor<B: Backend> {
    fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>);
}

pub trait ModuleVisitorMut<B: Backend> {
    fn visit_mut<const D: usize>(&mut self, id: &ParamId, tensor: &mut Tensor<B, D>);
}

/// Module with auto-differentiation backend.
pub trait ADModule:
    Module<Backend = Self::ADBackend> + Send + Sync + core::fmt::Debug + core::fmt::Display
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

impl core::fmt::Display for LoadingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("Loading error: {}", self.message).as_str())
    }
}

// TODO: Move from std to core after Error is core (see https://github.com/rust-lang/rust/issues/103765)
#[cfg(feature = "std")]
impl std::error::Error for LoadingError {}
