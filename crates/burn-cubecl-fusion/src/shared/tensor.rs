use std::hash::Hash;

use cubecl::{
    ir::{ElemType, ExpandElement, FloatKind, IntKind, StorageType, Type, UIntKind},
    prelude::*,
    unexpanded,
};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

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

#[derive(CubeType, Clone)]
pub enum GlobalScalar {
    F64(f64),
    F32(f32),
    F16(f16),
    BF16(bf16),
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
        let dtype = C::as_type(scope).elem_type();

        match self {
            GlobalScalarExpand::U64(val) => {
                if dtype == ElemType::UInt(cubecl::ir::UIntKind::U64) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::U32(val) => {
                if dtype == ElemType::UInt(cubecl::ir::UIntKind::U32) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::U16(val) => {
                if dtype == ElemType::UInt(cubecl::ir::UIntKind::U16) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::F64(val) => {
                if dtype == ElemType::Float(cubecl::ir::FloatKind::F64) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::F32(val) => {
                if dtype == ElemType::Float(cubecl::ir::FloatKind::F32) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::F16(val) => {
                if dtype == ElemType::Float(cubecl::ir::FloatKind::F16) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::BF16(val) => {
                if dtype == ElemType::Float(cubecl::ir::FloatKind::BF16) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::U8(val) => {
                if dtype == ElemType::UInt(cubecl::ir::UIntKind::U8) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }

            GlobalScalarExpand::I64(val) => {
                if dtype == ElemType::Int(cubecl::ir::IntKind::I64) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::I32(val) => {
                if dtype == ElemType::Int(cubecl::ir::IntKind::I32) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::I16(val) => {
                if dtype == ElemType::Int(cubecl::ir::IntKind::I16) {
                    let expand: ExpandElement = val.clone().into();
                    ExpandElementTyped::from(expand.clone())
                } else {
                    C::__expand_cast_from(scope, val.clone())
                }
            }
            GlobalScalarExpand::I8(val) => {
                if dtype == ElemType::Int(cubecl::ir::IntKind::I8) {
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
    type CompilationArg = GlobalScalarCompilationArg;

    fn compilation_arg<R: Runtime>(arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        match arg {
            GlobalScalar::F64(_) => {
                GlobalScalarCompilationArg::new(ElemType::Float(FloatKind::F64).into())
            }
            GlobalScalar::F32(_) => {
                GlobalScalarCompilationArg::new(ElemType::Float(FloatKind::F32).into())
            }
            GlobalScalar::F16(_) => {
                GlobalScalarCompilationArg::new(ElemType::Float(FloatKind::F16).into())
            }
            GlobalScalar::BF16(_) => {
                GlobalScalarCompilationArg::new(ElemType::Float(FloatKind::BF16).into())
            }
            GlobalScalar::I64(_) => {
                GlobalScalarCompilationArg::new(ElemType::Int(IntKind::I64).into())
            }
            GlobalScalar::I32(_) => {
                GlobalScalarCompilationArg::new(ElemType::Int(IntKind::I32).into())
            }
            GlobalScalar::I16(_) => {
                GlobalScalarCompilationArg::new(ElemType::Int(IntKind::I16).into())
            }
            GlobalScalar::I8(_) => {
                GlobalScalarCompilationArg::new(ElemType::Int(IntKind::I8).into())
            }
            GlobalScalar::U64(_) => {
                GlobalScalarCompilationArg::new(ElemType::UInt(UIntKind::U64).into())
            }
            GlobalScalar::U32(_) => {
                GlobalScalarCompilationArg::new(ElemType::UInt(UIntKind::U32).into())
            }
            GlobalScalar::U16(_) => {
                GlobalScalarCompilationArg::new(ElemType::UInt(UIntKind::U16).into())
            }
            GlobalScalar::U8(_) => {
                GlobalScalarCompilationArg::new(ElemType::UInt(UIntKind::U8).into())
            }
        }
    }

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        let expand = builder.scalar(arg.ty);
        match arg.ty.elem_type() {
            ElemType::Float(float_kind) => match float_kind {
                FloatKind::F16 => GlobalScalarExpand::F16(expand.into()),
                FloatKind::BF16 => GlobalScalarExpand::BF16(expand.into()),
                FloatKind::Flex32 => GlobalScalarExpand::F32(expand.into()),
                FloatKind::F32 => GlobalScalarExpand::F32(expand.into()),
                FloatKind::TF32 => GlobalScalarExpand::F32(expand.into()),
                FloatKind::F64 => GlobalScalarExpand::F32(expand.into()),
                FloatKind::E2M1
                | FloatKind::E2M3
                | FloatKind::E3M2
                | FloatKind::E4M3
                | FloatKind::E5M2
                | FloatKind::UE8M0 => unimplemented!("FP8 can't be passed as scalar"),
            },
            ElemType::Int(int_kind) => match int_kind {
                IntKind::I8 => GlobalScalarExpand::I8(expand.into()),
                IntKind::I16 => GlobalScalarExpand::I16(expand.into()),
                IntKind::I32 => GlobalScalarExpand::I32(expand.into()),
                IntKind::I64 => GlobalScalarExpand::I64(expand.into()),
            },
            ElemType::UInt(uint_kind) => match uint_kind {
                UIntKind::U8 => GlobalScalarExpand::U8(expand.into()),
                UIntKind::U16 => GlobalScalarExpand::U16(expand.into()),
                UIntKind::U32 => GlobalScalarExpand::U32(expand.into()),
                UIntKind::U64 => GlobalScalarExpand::U64(expand.into()),
            },
            ElemType::Bool => panic!("Bool should be converted first."),
        }
    }
}

#[derive(new, Serialize, Deserialize, Clone, PartialEq, Eq, Hash, Debug)]
pub struct GlobalScalarCompilationArg {
    ty: StorageType,
}

impl CompilationArg for GlobalScalarCompilationArg {}

impl<R: Runtime> ArgSettings<R> for GlobalScalar {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        match self {
            GlobalScalar::F64(val) => launcher.register_f64(*val),
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

impl CompilationArg for GlobalTensorCompilationArg {}
