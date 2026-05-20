use crate::engine::codegen::{DynElem, DynSize, DynVector};

use cubecl::{ir::Type, prelude::*};
use std::hash::Hash;

/// Represents a global tensor with the given [element type](ElemType).
///
/// # Warning
///
/// The `tensor` field type [Vector<NumericExpand<DYN_ELEM_ID>>] must be set using polyfill before
/// use.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct GlobalTensor {
    /// The global tensor type.
    pub tensor: OwnedTensor<DynVector>,
    /// The element type of the tensor.
    #[cube(comptime)]
    pub ty: Type,
    /// Whether the current tensor is logically broadcasted.
    #[cube(comptime)]
    pub broadcasted: bool,
}

// Everything below is to implement [LaunchArg].

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct GlobalTensorCompilationArg {
    tensor: TensorCompilationArg,
    ty: Type,
    broadcasted: bool,
}

#[derive(new, Debug)]
pub struct GlobalTensorArg<R: Runtime> {
    pub tensor: <OwnedTensor<DynVector> as LaunchArg>::RuntimeArg<R>,
    pub ty: Type,
    pub broadcasted: bool,
    pub address_type: AddressType,
}

impl LaunchArg for GlobalTensor {
    type RuntimeArg<R: Runtime> = GlobalTensorArg<R>;
    type CompilationArg = GlobalTensorCompilationArg;

    fn register<R: Runtime>(
        arg: Self::RuntimeArg<R>,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        launcher.with_scope(|scope| set_polyfill::expand::<DynElem, DynSize>(scope, arg.ty));
        let tensor = OwnedTensor::<DynVector>::register(arg.tensor, launcher);
        GlobalTensorCompilationArg {
            tensor,
            ty: arg.ty,
            broadcasted: arg.broadcasted,
        }
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> GlobalTensorExpand {
        set_polyfill::expand::<DynElem, DynSize>(&builder.scope, arg.ty);
        let tensor = OwnedTensor::expand(&arg.tensor, builder);
        GlobalTensorExpand {
            tensor,
            ty: arg.ty,
            broadcasted: arg.broadcasted,
        }
    }
}
