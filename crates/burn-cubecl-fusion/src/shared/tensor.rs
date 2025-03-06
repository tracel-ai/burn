use std::hash::Hash;

use cubecl::{
    ir::{Elem, ExpandElement, FloatKind, IntKind, Item, UIntKind},
    prelude::*,
    unexpanded,
};
use serde::{Deserialize, Serialize};

use super::DYN_ELEM_ID;

#[derive(CubeType)]
pub struct GlobalTensor {
    pub tensor: Tensor<Line<NumericExpand<DYN_ELEM_ID>>>,
    #[cube(comptime)]
    pub elem: Elem,
    #[cube(comptime)]
    pub broadcasted: bool,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Hash, Debug)]
pub struct GlobalTensorCompilationArg {
    tensor: TensorCompilationArg,
    elem: Elem,
    broadcasted: bool,
}

#[derive(new, Debug)]
pub struct GlobalTensorArg<'a, R: Runtime> {
    pub tensor: <Tensor<Line<NumericExpand<DYN_ELEM_ID>>> as LaunchArg>::RuntimeArg<'a, R>,
    pub elem: Elem,
    pub broadcasted: bool,
}

#[derive(CubeType)]
pub enum GlobalScalar {
    F32(f32),
    F16(half::f16),
    BF16(half::bf16),
    I64(i64),
    I32(i32),
    I16(i16),
    I8(i8),
    U64(u64),
    U32(u32),
    U16(u16),
    U8(u8),
}

impl GlobalScalar {
    pub fn as_u32(&self) -> u32 {
        unexpanded!()
    }

    pub fn read<C: CubePrimitive>(&self) -> C {
        unexpanded!()
    }
}

impl GlobalScalarExpand {
    pub fn __expand_as_u32_method(&self, _scope: &mut Scope) -> ExpandElementTyped<u32> {
        match self {
            GlobalScalarExpand::U32(val) => val.clone(),
            _ => todo!(),
        }
    }
    pub fn __expand_read_method<C: CubePrimitive>(
        &self,
        scope: &mut Scope,
    ) -> ExpandElementTyped<C> {
        let dtype = C::as_elem(scope);

        match self {
            GlobalScalarExpand::U64(val) => {
                if dtype == Elem::UInt(cubecl::ir::UIntKind::U64) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::U32(val) => {
                if dtype == Elem::UInt(cubecl::ir::UIntKind::U32) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::U16(val) => {
                if dtype == Elem::UInt(cubecl::ir::UIntKind::U16) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::F32(val) => {
                if dtype == Elem::Float(cubecl::ir::FloatKind::F32) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::F16(val) => {
                if dtype == Elem::Float(cubecl::ir::FloatKind::F16) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::BF16(val) => {
                if dtype == Elem::Float(cubecl::ir::FloatKind::BF16) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::U8(val) => {
                if dtype == Elem::UInt(cubecl::ir::UIntKind::U8) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }

            GlobalScalarExpand::I64(val) => {
                if dtype == Elem::Int(cubecl::ir::IntKind::I64) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::I32(val) => {
                if dtype == Elem::Int(cubecl::ir::IntKind::I32) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::I16(val) => {
                if dtype == Elem::Int(cubecl::ir::IntKind::I16) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::I8(val) => {
                if dtype == Elem::Int(cubecl::ir::IntKind::I8) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
        }
    }
}

impl LaunchArg for GlobalScalar {
    type RuntimeArg<'a, R: Runtime> = GlobalScalar;

    fn compilation_arg<R: Runtime>(arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        match arg {
            GlobalScalar::F32(_) => GlobalScalarCompilationArg::new(Elem::Float(FloatKind::F32)),
            GlobalScalar::F16(_) => GlobalScalarCompilationArg::new(Elem::Float(FloatKind::F16)),
            GlobalScalar::BF16(_) => GlobalScalarCompilationArg::new(Elem::Float(FloatKind::BF16)),
            GlobalScalar::I64(_) => GlobalScalarCompilationArg::new(Elem::Int(IntKind::I64)),
            GlobalScalar::I32(_) => GlobalScalarCompilationArg::new(Elem::Int(IntKind::I32)),
            GlobalScalar::I16(_) => GlobalScalarCompilationArg::new(Elem::Int(IntKind::I16)),
            GlobalScalar::I8(_) => GlobalScalarCompilationArg::new(Elem::Int(IntKind::I8)),
            GlobalScalar::U64(_) => GlobalScalarCompilationArg::new(Elem::UInt(UIntKind::U64)),
            GlobalScalar::U32(_) => GlobalScalarCompilationArg::new(Elem::UInt(UIntKind::U32)),
            GlobalScalar::U16(_) => GlobalScalarCompilationArg::new(Elem::UInt(UIntKind::U16)),
            GlobalScalar::U8(_) => GlobalScalarCompilationArg::new(Elem::UInt(UIntKind::U8)),
        }
    }
}

#[derive(new, Serialize, Deserialize, Clone, PartialEq, Eq, Hash, Debug)]
pub struct GlobalScalarCompilationArg {
    elem: Elem,
}

impl CompilationArg for GlobalScalarCompilationArg {}

impl LaunchArgExpand for GlobalScalar {
    type CompilationArg = GlobalScalarCompilationArg;

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        let expand = builder.scalar(arg.elem);
        match arg.elem {
            Elem::Float(float_kind) | Elem::AtomicFloat(float_kind) => match float_kind {
                FloatKind::F16 => GlobalScalarExpand::F16(expand.into()),
                FloatKind::BF16 => GlobalScalarExpand::BF16(expand.into()),
                FloatKind::Flex32 => GlobalScalarExpand::F32(expand.into()),
                FloatKind::F32 => GlobalScalarExpand::F32(expand.into()),
                FloatKind::TF32 => GlobalScalarExpand::F32(expand.into()),
                FloatKind::F64 => GlobalScalarExpand::F32(expand.into()),
            },
            Elem::Int(int_kind) | Elem::AtomicInt(int_kind) => match int_kind {
                IntKind::I8 => GlobalScalarExpand::I8(expand.into()),
                IntKind::I16 => GlobalScalarExpand::I16(expand.into()),
                IntKind::I32 => GlobalScalarExpand::I32(expand.into()),
                IntKind::I64 => GlobalScalarExpand::I64(expand.into()),
            },
            Elem::UInt(uint_kind) | Elem::AtomicUInt(uint_kind) => match uint_kind {
                UIntKind::U8 => GlobalScalarExpand::U8(expand.into()),
                UIntKind::U16 => GlobalScalarExpand::U16(expand.into()),
                UIntKind::U32 => GlobalScalarExpand::U32(expand.into()),
                UIntKind::U64 => GlobalScalarExpand::U64(expand.into()),
            },
            Elem::Bool => panic!("Bool should be converted first."),
        }
    }
}

impl<R: Runtime> ArgSettings<R> for GlobalScalar {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        match self {
            GlobalScalar::F32(val) => launcher.register_f32(*val),
            GlobalScalar::F16(val) => launcher.register_f16(*val),
            GlobalScalar::BF16(val) => launcher.register_bf16(*val),
            GlobalScalar::I64(val) => launcher.register_i64(*val),
            GlobalScalar::I32(val) => launcher.register_i32(*val),
            GlobalScalar::I16(val) => launcher.register_i16(*val),
            GlobalScalar::I8(val) => launcher.register_i8(*val),
            GlobalScalar::U64(val) => launcher.register_u64(*val),
            GlobalScalar::U32(val) => launcher.register_u32(*val),
            GlobalScalar::U16(val) => launcher.register_u16(*val),
            GlobalScalar::U8(val) => launcher.register_u8(*val),
        }
    }
}

impl LaunchArg for GlobalTensor {
    type RuntimeArg<'a, R: Runtime> = GlobalTensorArg<'a, R>;

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
}

impl<R: Runtime> ArgSettings<R> for GlobalTensorArg<'_, R> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        launcher.register_tensor(&self.tensor)
    }
}

impl CompilationArg for GlobalTensorCompilationArg {}

impl LaunchArgExpand for GlobalTensor {
    type CompilationArg = GlobalTensorCompilationArg;

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> GlobalTensorExpand {
        let tensor = builder.input_tensor(Item::vectorized(arg.elem, arg.tensor.vectorisation));

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
            None => builder.output_tensor(Item::vectorized(arg.elem, arg.tensor.vectorisation)),
        };
        GlobalTensorExpand {
            tensor: tensor.into(),
            elem: arg.elem,
            broadcasted: arg.broadcasted,
        }
    }
}
