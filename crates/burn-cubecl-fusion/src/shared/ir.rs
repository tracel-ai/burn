use burn_tensor::DType;
use burn_tensor::quantization::{QuantScheme, QuantStore, QuantValue};
use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, UIntKind};
use cubecl::prelude::*;
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

use super::tensor::{GlobalScalar, GlobalTensor};

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
/// Argument to a [fuse operation](FuseOp).
pub enum Arg {
    Input(u32, FusePrecision, LayoutInfo),
    Local(u32, FusePrecision),
    Output(u32, FusePrecision, LayoutInfo),
    Scalar(u32, FusePrecision),
    ScalarShape(u32),
    /// Only constant that can be encoded into an u32 can be used as literal.
    Literal(u32, FusePrecision),
    InputReshaped {
        original: Box<Arg>,
        shape: Sequence<Arg>,
        broadcasted: bool,
    },
    InputSwapDims {
        original: Box<Arg>,
        dims: (u32, u32),
        broadcasted: bool,
    },
}

#[derive(
    CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord,
)]
/// Layout information.
pub enum LayoutInfo {
    /// The layout if the same as the reference.
    SameAsRef,
    /// The reference layout.
    IsRef,
    /// The layout if unknown.
    Unknown,
}

impl Arg {
    pub fn precision(&self) -> FusePrecision {
        *match self {
            Arg::Input(_, p, _) => p,
            Arg::Local(_, p) => p,
            Arg::Output(_, p, _) => p,
            Arg::Scalar(_, p) => p,
            Arg::Literal(_, p) => p,
            Arg::ScalarShape(_) => return FusePrecision::U32,
            Arg::InputReshaped { original, .. } => return original.precision(),
            Arg::InputSwapDims { original, .. } => return original.precision(),
        }
    }
}

impl CubeType for Arg {
    type ExpandType = Self;
}

impl IntoMut for Arg {
    fn into_mut(self, _context: &mut Scope) -> Self {
        self
    }
}

impl IntoRuntime for Arg {
    fn __expand_runtime_method(self, _context: &mut Scope) -> Self::ExpandType {
        self
    }
}

impl CubeDebug for Arg {}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// Operations that can be executed and fused automatically using a fuse-on-read and/or
/// fuse-on-write strategy.
pub enum FuseOp {
    Add(BinaryFuseArgs),
    Sub(BinaryFuseArgs),
    Mul(BinaryFuseArgs),
    Div(BinaryFuseArgs),
    Powf(BinaryFuseArgs),
    Abs(UnaryFuseArgs),
    Exp(UnaryFuseArgs),
    Log(UnaryFuseArgs),
    Log1p(UnaryFuseArgs),
    Cos(UnaryFuseArgs),
    Sin(UnaryFuseArgs),
    Tanh(UnaryFuseArgs),
    Erf(UnaryFuseArgs),
    Sqrt(UnaryFuseArgs),
    Recip(UnaryFuseArgs),
    Assign(UnaryFuseArgs),
    Equal(BinaryFuseArgs),
    Lower(BinaryFuseArgs),
    Greater(BinaryFuseArgs),
    LowerEqual(BinaryFuseArgs),
    Rem(BinaryFuseArgs),
    GreaterEqual(BinaryFuseArgs),
    Clamp {
        input: Arg,
        min: Arg,
        max: Arg,
        out: Arg,
    },
    ConditionalAssign {
        cond: Arg,
        lhs: Arg,
        rhs: Arg,
        out: Arg,
    },
    Gather {
        input: Arg,
        indices: Arg,
        output: Arg,
        dim: u32,
    },
    Select {
        input: Arg,
        indices: Arg,
        output: Arg,
        dim: u32,
    },
    Dequantize {
        values: Arg,
        params: Arg,
        output: Arg,
        scheme: QuantSchemeFuse,
    },
}

#[derive(
    CubeType, CubeLaunch, Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord,
)]
pub struct QuantSchemeFuse {
    #[cube(comptime)]
    pub(crate) scheme: QuantScheme,
}

impl FuseOp {
    /// Element type used for the computation.
    pub(crate) fn cmp_elem(&self) -> ElemType {
        match self {
            FuseOp::Add(op) => op.lhs.precision().into_elem(),
            FuseOp::Sub(op) => op.lhs.precision().into_elem(),
            FuseOp::Mul(op) => op.lhs.precision().into_elem(),
            FuseOp::Div(op) => op.lhs.precision().into_elem(),
            FuseOp::Powf(op) => op.lhs.precision().into_elem(),
            FuseOp::Abs(op) => op.out.precision().into_elem(),
            FuseOp::Exp(op) => op.out.precision().into_elem(),
            FuseOp::Log(op) => op.out.precision().into_elem(),
            FuseOp::Log1p(op) => op.out.precision().into_elem(),
            FuseOp::Cos(op) => op.out.precision().into_elem(),
            FuseOp::Sin(op) => op.out.precision().into_elem(),
            FuseOp::Tanh(op) => op.out.precision().into_elem(),
            FuseOp::Erf(op) => op.out.precision().into_elem(),
            FuseOp::Recip(op) => op.out.precision().into_elem(),
            FuseOp::Sqrt(op) => op.out.precision().into_elem(),
            FuseOp::Assign(op) => op.out.precision().into_elem(),
            FuseOp::Equal(op) => op.lhs.precision().into_elem(),
            FuseOp::Lower(op) => op.lhs.precision().into_elem(),
            FuseOp::Greater(op) => op.lhs.precision().into_elem(),
            FuseOp::LowerEqual(op) => op.lhs.precision().into_elem(),
            FuseOp::GreaterEqual(op) => op.lhs.precision().into_elem(),
            FuseOp::ConditionalAssign { out, .. } => out.precision().into_elem(),
            FuseOp::Gather { output, .. } => output.precision().into_elem(),
            FuseOp::Select { output, .. } => output.precision().into_elem(),
            FuseOp::Dequantize { output, .. } => output.precision().into_elem(),
            FuseOp::Rem(op) => op.out.precision().into_elem(),
            FuseOp::Clamp { out, .. } => out.precision().into_elem(),
        }
    }

    /// Element type used for the computation.
    pub(crate) fn cmp_type(&self) -> StorageType {
        self.cmp_elem().into()
    }
}

#[derive(CubeType, CubeLaunch, Default, Clone)]
/// Global arguments that are used for fusing [element wise operations](ElemTypewiseOp).
pub struct GlobalArgs {
    pub tensors: Sequence<GlobalTensor>,
    pub scalars: Sequence<GlobalScalar>,
    pub reshapes: Sequence<u32>,
}

impl GlobalArgsExpand {
    pub fn __expand_clone_method(&self, _scope: &mut Scope) -> Self {
        self.clone()
    }
}

impl<R: Runtime> Default for GlobalArgsLaunch<'_, R> {
    fn default() -> Self {
        Self {
            tensors: Default::default(),
            scalars: Default::default(),
            reshapes: Default::default(),
            _phantom_runtime: std::marker::PhantomData,
            _phantom_a: std::marker::PhantomData,
        }
    }
}

impl<R: Runtime> core::fmt::Debug for GlobalArgsLaunch<'_, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?})", self.tensors.values)
    }
}

impl<R: Runtime> GlobalArgsLaunch<'_, R> {
    /// Get the shape of the given [argument](Arg).
    ///
    /// # Panics
    ///
    /// If the argument doesn't have an handle.
    pub fn shape(&self, arg: &Arg) -> Vec<usize> {
        match self.resolve_arg(arg) {
            TensorArg::Handle { handle, .. } => handle.shape.to_vec(),
            TensorArg::Alias { .. } => panic!("Unsupported yet"),
        }
    }

    /// Shape used by the reference tensor.
    pub fn shape_ref(&self, ref_layout: &RefLayout, rank: usize) -> Vec<usize> {
        match ref_layout {
            RefLayout::Concrete(arg) => self.shape(arg),
            RefLayout::Virtual(layout) => match layout {
                VirtualLayout::SwapDims(original, dims) => {
                    let mut shape = self.shape(original);
                    shape.swap(dims.0 as usize, dims.1 as usize);
                    shape
                }
                VirtualLayout::Reshaped { reshape_pos, .. } => {
                    let start = *reshape_pos as usize * rank;
                    let end = start + rank;
                    self.reshapes.values[start..end]
                        .iter()
                        .map(|s| s.elem as usize)
                        .collect()
                }
                VirtualLayout::Shape(original, _) => self.shape(original),
            },
        }
    }

    /// Get the strides of the given [argument](Arg).
    ///
    /// # Panics
    ///
    /// If the argument doesn't have an handle.
    pub fn strides(&self, arg: &Arg) -> Vec<usize> {
        match self.resolve_arg(arg) {
            TensorArg::Handle { handle, .. } => handle.strides.to_vec(),
            TensorArg::Alias { .. } => panic!("Unsupported yet"),
        }
    }

    pub fn strides_ref(&self, ref_layout: &RefLayout, rank: usize) -> Vec<usize> {
        match ref_layout {
            RefLayout::Concrete(arg) => self.strides(arg),
            // When not concrete, we operate on the contiguous layout.
            _ => {
                let shape = self.shape_ref(ref_layout, rank);
                let mut strides = vec![0; shape.len()];

                let mut current = 1;
                shape.iter().enumerate().rev().for_each(|(index, val)| {
                    strides[index] = current;
                    current *= val;
                });

                strides
            }
        }
    }

    /// Get the line size of the given [argument](Arg).
    ///
    /// # Panics
    ///
    /// If the argument doesn't have an handle.
    pub fn line_size(&self, arg: &Arg) -> u8 {
        match self.resolve_arg(arg) {
            TensorArg::Handle { line_size, .. } => *line_size,
            TensorArg::Alias { .. } => panic!("Unsupported yet"),
        }
    }

    /// Resolve the [argument](Arg) to a [tensor argument](TensorArg).
    ///
    /// # Panics
    ///
    /// If the argument isn't a global input or output tensor.
    pub fn resolve_arg(&self, arg: &Arg) -> &TensorArg<'_, R> {
        match arg {
            Arg::Input(pos, _, _) => &self.tensors.values[*pos as usize].tensor,
            Arg::Output(pos, _, _) => &self.tensors.values[*pos as usize].tensor,
            other => panic!("Arg not found: {other:?}"),
        }
    }
}

#[derive(CubeType, Clone)]
/// Keep track of all local variables that are used as argument in fused
/// [element wise operations](ElemwiseOp).
pub struct LocalArgs {
    pub l_f64: Registry<u32, Line<f64>>,
    pub l_f32: Registry<u32, Line<f32>>,
    pub l_f16: Registry<u32, Line<f16>>,
    pub l_bf16: Registry<u32, Line<bf16>>,
    pub l_i64: Registry<u32, Line<i64>>,
    pub l_i32: Registry<u32, Line<i32>>,
    pub l_i16: Registry<u32, Line<i16>>,
    pub l_i8: Registry<u32, Line<i8>>,
    pub l_u64: Registry<u32, Line<u64>>,
    pub l_u32: Registry<u32, Line<u32>>,
    pub l_u16: Registry<u32, Line<u16>>,
    pub l_u8: Registry<u32, Line<u8>>,
    pub l_bool: Registry<u32, Line<bool>>,
    pub ref_shape: Slice<u32>,
    pub ref_strides: Slice<u32>,
    #[cube(comptime)]
    pub ref_line_size: u32,
}

#[cube]
impl LocalArgs {
    pub fn new(
        ref_shape: Slice<u32>,
        ref_strides: Slice<u32>,
        #[comptime] ref_line_size: u32,
    ) -> LocalArgs {
        LocalArgs {
            l_f64: Registry::<u32, Line<f64>>::new(),
            l_f32: Registry::<u32, Line<f32>>::new(),
            l_f16: Registry::<u32, Line<f16>>::new(),
            l_bf16: Registry::<u32, Line<bf16>>::new(),
            l_i64: Registry::<u32, Line<i64>>::new(),
            l_i32: Registry::<u32, Line<i32>>::new(),
            l_i16: Registry::<u32, Line<i16>>::new(),
            l_i8: Registry::<u32, Line<i8>>::new(),
            l_u64: Registry::<u32, Line<u64>>::new(),
            l_u32: Registry::<u32, Line<u32>>::new(),
            l_u16: Registry::<u32, Line<u16>>::new(),
            l_u8: Registry::<u32, Line<u8>>::new(),
            l_bool: Registry::<u32, Line<bool>>::new(),
            ref_shape,
            ref_strides,
            ref_line_size,
        }
    }
}

impl LocalArgsExpand {
    pub fn __expand_clone_method(&self, _scope: &mut Scope) -> Self {
        self.clone()
    }
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// Unary [element wise operation](ElemwiseOp) arguments.
pub struct UnaryFuseArgs {
    pub input: Arg,
    pub out: Arg,
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// Binary [element wise operation](ElemwiseOp) arguments.
pub struct BinaryFuseArgs {
    pub lhs: Arg,
    pub rhs: Arg,
    pub out: Arg,
}

#[derive(
    CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
/// Precisions supported by [element wise operations](ElemwiseOp).
pub enum FusePrecision {
    F64,
    F32,
    Flex32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
}

impl From<ElemType> for FusePrecision {
    fn from(value: ElemType) -> Self {
        match value {
            ElemType::Float(kind) => match kind {
                FloatKind::F16 => Self::F16,
                FloatKind::BF16 => Self::BF16,
                FloatKind::F32 => Self::F32,
                FloatKind::Flex32 => Self::Flex32,
                _ => panic!("Unsupported precision for fusion: {value}"),
            },
            ElemType::Int(kind) => match kind {
                IntKind::I64 => Self::I64,
                IntKind::I32 => Self::I32,
                IntKind::I16 => Self::I16,
                IntKind::I8 => Self::I8,
            },
            ElemType::UInt(kind) => match kind {
                UIntKind::U64 => Self::U64,
                UIntKind::U32 => Self::U32,
                UIntKind::U16 => Self::U16,
                UIntKind::U8 => Self::U8,
            },
            ElemType::Bool => Self::Bool,
        }
    }
}

impl From<StorageType> for FusePrecision {
    fn from(value: StorageType) -> Self {
        value.elem_type().into()
    }
}

impl FusePrecision {
    pub fn into_elem(self) -> ElemType {
        match self {
            FusePrecision::F32 => ElemType::Float(FloatKind::F32),
            FusePrecision::Flex32 => ElemType::Float(FloatKind::Flex32),
            FusePrecision::F16 => ElemType::Float(FloatKind::F16),
            FusePrecision::BF16 => ElemType::Float(FloatKind::BF16),
            FusePrecision::I64 => ElemType::Int(IntKind::I64),
            FusePrecision::I32 => ElemType::Int(IntKind::I32),
            FusePrecision::I16 => ElemType::Int(IntKind::I16),
            FusePrecision::I8 => ElemType::Int(IntKind::I8),
            FusePrecision::U64 => ElemType::UInt(UIntKind::U64),
            FusePrecision::U32 => ElemType::UInt(UIntKind::U32),
            FusePrecision::U16 => ElemType::UInt(UIntKind::U16),
            FusePrecision::U8 => ElemType::UInt(UIntKind::U8),
            FusePrecision::Bool => ElemType::Bool,
            FusePrecision::F64 => ElemType::Float(FloatKind::F64),
        }
    }

    pub fn into_type(self) -> StorageType {
        self.into_elem().into()
    }
}

impl From<DType> for FusePrecision {
    fn from(value: DType) -> Self {
        match value {
            DType::F32 => Self::F32,
            DType::Flex32 => Self::Flex32,
            DType::F16 => Self::F16,
            DType::BF16 => Self::BF16,
            DType::I64 => Self::I64,
            DType::I32 => Self::I32,
            DType::I16 => Self::I16,
            DType::I8 => Self::I8,
            DType::U64 => Self::U64,
            DType::U32 => Self::U32,
            DType::U16 => Self::U16,
            DType::U8 => Self::U8,
            DType::Bool => Self::Bool,
            DType::F64 => Self::F64,
            DType::QFloat(scheme) => match scheme.store {
                QuantStore::Native => match scheme.value {
                    QuantValue::Q8F | QuantValue::Q8S => Self::I8,
                    QuantValue::E4M3 | QuantValue::E5M2 | QuantValue::E2M1 => {
                        unimplemented!("Unsupported precision for fusion")
                    }
                    QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S => {
                        panic!("Can't store native sub-byte values")
                    }
                },
                QuantStore::U32 => Self::U32,
            },
            DType::Complex64 => todo!(),
            DType::Complex32 => todo!(),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// Configuration that encapsulates all comptime information necessary for element wise fusion.
pub struct FuseBlockConfig {
    pub rank: u32,
    pub ref_layout: RefLayout,
    pub ops: Sequence<FuseOp>,
    pub width: u8,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// A reference layout determines how a fuse execution will access elements in tensors.
///
/// It can either follow the same layout as a concrete tensor, or follow a virtual layout.
pub enum RefLayout {
    Concrete(Arg),
    Virtual(VirtualLayout),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// A virtual layout is always contiguous and retrieve its shape from either a reshape tensor or a
/// tensor with swap dimensions.
pub enum VirtualLayout {
    /// Virtual tensor with the provided shape id and contiguous strides.
    Reshaped { reshape_pos: u32, line_size: u32 },
    /// Virtual tensor with the same shape as the given input, but with swap dims and contiguous
    /// strides.
    SwapDims(Arg, (u32, u32)),
    /// Virtual tensor with the same shape as the given input, but with contiguous strides.
    Shape(Arg, u32),
}

impl Arg {
    /// Add layout information; it's going to impact how the input or output is read
    /// and written to.
    pub fn add_layout_info(&mut self, layout: LayoutInfo) {
        match self {
            Arg::Input(_, _, old) => {
                *old = layout;
            }
            Arg::Output(_, _, old) => {
                *old = layout;
            }
            _ => {}
        }
    }
}

impl RegistryQuery<Self> for Arg {}
