use crate::engine::codegen::{DynElem, DynSize};

use cubecl::{ir::Type, prelude::*};
use serde::{Deserialize, Serialize};
use std::hash::Hash;

/// Represents a global tensor with the given [element type](ElemType).
///
/// # Warning
///
/// The `tensor` field type [Vector<NumericExpand<DYN_ELEM_ID>>] must be set using polyfill before
/// use.
#[derive(CubeType, Clone)]
pub struct GlobalTensor {
    /// The global tensor type.
    pub tensor: Tensor<Vector<DynElem, DynSize>>,
    /// The element type of the tensor.
    #[cube(comptime)]
    pub ty: Type,
    /// Whether the current tensor is logically broadcasted.
    #[cube(comptime)]
    pub broadcasted: bool,
}

// Everything below is to implement [LaunchArg].

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash, Debug)]
pub struct GlobalTensorCompilationArg {
    tensor: TensorCompilationArg,
    ty: Type,
    broadcasted: bool,
}

#[derive(new, Debug)]
pub struct GlobalTensorArg<R: Runtime> {
    pub tensor: <Tensor<Vector<DynElem, DynSize>> as LaunchArg>::RuntimeArg<R>,
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
        let tensor = TensorCompilationArg {
            inplace: match arg.tensor {
                TensorArg::Handle { .. } => None,
                TensorArg::Alias { input_pos, .. } => Some(input_pos as u32),
            },
        };
        launcher.register_tensor(arg.tensor, arg.ty);

        GlobalTensorCompilationArg {
            tensor,
            ty: arg.ty,
            broadcasted: arg.broadcasted,
        }
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> GlobalTensorExpand {
        let tensor = builder.input_tensor(arg.ty);

        GlobalTensorExpand {
            tensor: tensor.into(),
            ty: arg.ty,
            broadcasted: arg.broadcasted,
        }
    }

    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> GlobalTensorExpand {
        let tensor = match arg.tensor.inplace {
            Some(id) => builder.inplace_output(id),
            None => builder.output_tensor(arg.ty),
        };
        GlobalTensorExpand {
            tensor: tensor.into(),
            ty: arg.ty,
            broadcasted: arg.broadcasted,
        }
    }
}
