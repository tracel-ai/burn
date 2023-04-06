use alloc::vec::Vec;

use super::ParamId;
use crate::{
    record::Record,
    tensor::backend::{ADBackend, Backend},
};
pub use burn_derive::Module;
use burn_tensor::Tensor;

macro_rules! module {
    (map=$module:ident, ops=$item:expr) => {{
        struct Mapper;
        impl<B: Backend> ModuleMapper<B> for Mapper {
            fn map<const D: usize>(&mut self, _id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
                let func = $item;
                func(tensor)
            }
        }
        let mut mapper = Mapper;
        $module.map(&mut mapper)
    }};
    (map=$module:ident, ops=$item:expr, capture={$capture:ident: $ty:ty}) => {{
        struct Mapper<'a, B: Backend> {
            capture: &'a $ty,
            backend: core::marker::PhantomData<B>,
        }
        impl<'a, B: Backend> ModuleMapper<B> for Mapper<'a, B> {
            fn map<const D: usize>(&mut self, _id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
                let func = $item;
                func(tensor, self.capture)
            }
        }
        let mut mapper = Mapper {
            capture: $capture,
            backend: core::marker::PhantomData::default(),
        };
        $module.map(&mut mapper)
    }};
    (visit=$module:ident, ops=$item:expr, state=$state_ty:ty, init=$init:expr) => {{
        struct Visitor<'a, B: Backend> {
            state: &'a mut $state_ty,
            backend: core::marker::PhantomData<B>,
        }
        impl<'a, B: Backend> ModuleVisitor<B> for Visitor<'a, B> {
            fn visit<const D: usize>(&mut self, _id: &ParamId, tensor: &Tensor<B, D>) {
                let func = $item;
                func(tensor, &mut self.state)
            }
        }
        let mut state = $init();
        let mut visitor = Visitor {
            state: &mut state,
            backend: core::marker::PhantomData::default(),
        };
        $module.visit(&mut visitor);
        state
    }};
}

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
    /// Type to save and load the module.
    type Record: Record;

    /// Get the device list of the module and all of its sub-modules.
    fn devices(&self) -> Vec<B::Device> {
        module!(
            visit = self,
            ops = |tensor: &Tensor<B, D>, state: &mut Vec<B::Device>| {
                let device = tensor.device();
                if !state.contains(&device) {
                    state.push(device);
                }
            },
            state = Vec<B::Device>,
            init = Vec::new
        )
    }
    /// Move the module and all of its sub-modules to the given device.
    fn to_device(self, device: &B::Device) -> Self {
        println!("To device");
        module!(
            map = self,
            ops =
                |tensor: Tensor<B, D>, device: &B::Device| tensor.to_device(device).require_grad(),
            capture = { device: B::Device }
        )
    }
    /// Detach the module from the graph.
    fn detach(self) -> Self {
        module!(map = self, ops = Tensor::detach)
    }
    /// Mark each tensor in the module tree as tracked.
    fn require_grad(self) -> Self {
        module!(map = self, ops = Tensor::require_grad)
    }
    /// Get the number of parameters the module has, including all of its sub-modules.
    fn num_params(&self) -> usize {
        module!(
            visit = self,
            ops = |tensor: &Tensor<B, D>, state: &mut usize| {
                *state += tensor.shape().num_elements();
            },
            state = usize,
            init = || 0
        )
    }
    /// Visit each tensor in the module with a [visitor](ModuleVisitor).
    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V);
    /// Map each tensor in the module with a [mapper](ModuleMapper).
    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self;
    /// Load the module state from a record.
    fn load_record(self, record: Self::Record) -> Self;
    /// Convert the module into a record containing the state.
    fn into_record(self) -> Self::Record;
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
