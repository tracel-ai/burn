use cubecl::{
    ir::{ElemType, Type},
    prelude::*,
};
use serde::{Deserialize, Serialize};
use std::hash::Hash;

use super::DYN_ELEM_ID;

#[derive(CubeType, Clone)]
pub struct GlobalTensor {
    pub tensor: Tensor<Line<NumericExpand<DYN_ELEM_ID>>>,
    #[cube(comptime)]
    pub elem: ElemType,
    #[cube(comptime)]
    pub broadcasted: bool,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash, Debug)]
pub struct GlobalTensorCompilationArg {
    tensor: TensorCompilationArg,
    elem: ElemType,
    broadcasted: bool,
}

#[derive(new, Debug)]
pub struct GlobalTensorArg<'a, R: Runtime> {
    pub tensor: <Tensor<Line<NumericExpand<DYN_ELEM_ID>>> as LaunchArg>::RuntimeArg<'a, R>,
    pub elem: ElemType,
    pub broadcasted: bool,
}

impl CompilationArg for GlobalTensorCompilationArg {}

impl LaunchArg for GlobalTensor {
    type RuntimeArg<'a, R: Runtime> = GlobalTensorArg<'a, R>;
    type CompilationArg = GlobalTensorCompilationArg;

    fn compilation_arg<R: Runtime>(runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        let tensor = <Tensor<Line<NumericExpand<DYN_ELEM_ID>>> as LaunchArg>::compilation_arg(
            &runtime_arg.tensor,
        );
        GlobalTensorCompilationArg {
            tensor,
            elem: runtime_arg.elem,
            broadcasted: runtime_arg.broadcasted,
        }
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> GlobalTensorExpand {
        let tensor = builder.input_tensor(Type::scalar(arg.elem).line(arg.tensor.line_size));

        GlobalTensorExpand {
            tensor: tensor.into(),
            elem: arg.elem,
            broadcasted: arg.broadcasted,
        }
    }
    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> GlobalTensorExpand {
        let tensor = match arg.tensor.inplace {
            Some(id) => builder.inplace_output(id),
            None => builder.output_tensor(Type::scalar(arg.elem).line(arg.tensor.line_size)),
        };
        GlobalTensorExpand {
            tensor: tensor.into(),
            elem: arg.elem,
            broadcasted: arg.broadcasted,
        }
    }
}

impl<R: Runtime> ArgSettings<R> for GlobalTensorArg<'_, R> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        launcher.register_tensor(&self.tensor)
    }
}
