use burn_tensor::DType;
use cubecl::ir::Elem;
use cubecl::prelude::*;
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

use super::{
    tensor::{GlobalScalar, GlobalTensor},
    DYN_ELEM_ID,
};

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
/// Argument to an [elemwise operation](ElemwiseOp).
pub enum Arg {
    Input(u32, ElemwisePrecision, LayoutInfo),
    Local(u32, ElemwisePrecision),
    Output(u32, ElemwisePrecision, LayoutInfo),
    Scalar(u32, ElemwisePrecision),
    ScalarShape(u32),
    /// Only constant that can be encoded into an u32 can be used as literal.
    Literal(u32, ElemwisePrecision),
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

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
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
    pub fn precision(&self) -> ElemwisePrecision {
        *match self {
            Arg::Input(_, p, _) => p,
            Arg::Local(_, p) => p,
            Arg::Output(_, p, _) => p,
            Arg::Scalar(_, p) => p,
            Arg::Literal(_, p) => p,
            Arg::ScalarShape(_) => return ElemwisePrecision::U32,
            Arg::InputReshaped { original, .. } => return original.precision(),
            Arg::InputSwapDims { original, .. } => return original.precision(),
        }
    }
}

impl CubeType for Arg {
    type ExpandType = Self;
}

impl Init for Arg {
    fn init(self, _context: &mut Scope) -> Self {
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
/// Operations that can be executed and fused.
pub enum ElemwiseOp {
    Add(BinaryElemwiseArgs),
    Sub(BinaryElemwiseArgs),
    Mul(BinaryElemwiseArgs),
    Div(BinaryElemwiseArgs),
    Powf(BinaryElemwiseArgs),
    Abs(UnaryElemwiseArgs),
    Exp(UnaryElemwiseArgs),
    Log(UnaryElemwiseArgs),
    Log1p(UnaryElemwiseArgs),
    Cos(UnaryElemwiseArgs),
    Sin(UnaryElemwiseArgs),
    Tanh(UnaryElemwiseArgs),
    Erf(UnaryElemwiseArgs),
    Recip(UnaryElemwiseArgs),
    Assign(UnaryElemwiseArgs),
    Equal(BinaryElemwiseArgs),
    Lower(BinaryElemwiseArgs),
    Greater(BinaryElemwiseArgs),
    LowerEqual(BinaryElemwiseArgs),
    GreaterEqual(BinaryElemwiseArgs),
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
}

impl ElemwiseOp {
    /// Element type used for the computation.
    pub(crate) fn cmp_elem(&self) -> Elem {
        match self {
            ElemwiseOp::Add(op) => op.lhs.precision().into_elem(),
            ElemwiseOp::Sub(op) => op.lhs.precision().into_elem(),
            ElemwiseOp::Mul(op) => op.lhs.precision().into_elem(),
            ElemwiseOp::Div(op) => op.lhs.precision().into_elem(),
            ElemwiseOp::Powf(op) => op.lhs.precision().into_elem(),
            ElemwiseOp::Abs(op) => op.out.precision().into_elem(),
            ElemwiseOp::Exp(op) => op.out.precision().into_elem(),
            ElemwiseOp::Log(op) => op.out.precision().into_elem(),
            ElemwiseOp::Log1p(op) => op.out.precision().into_elem(),
            ElemwiseOp::Cos(op) => op.out.precision().into_elem(),
            ElemwiseOp::Sin(op) => op.out.precision().into_elem(),
            ElemwiseOp::Tanh(op) => op.out.precision().into_elem(),
            ElemwiseOp::Erf(op) => op.out.precision().into_elem(),
            ElemwiseOp::Recip(op) => op.out.precision().into_elem(),
            ElemwiseOp::Assign(op) => op.out.precision().into_elem(),
            ElemwiseOp::Equal(op) => op.lhs.precision().into_elem(),
            ElemwiseOp::Lower(op) => op.lhs.precision().into_elem(),
            ElemwiseOp::Greater(op) => op.lhs.precision().into_elem(),
            ElemwiseOp::LowerEqual(op) => op.lhs.precision().into_elem(),
            ElemwiseOp::GreaterEqual(op) => op.lhs.precision().into_elem(),
            ElemwiseOp::ConditionalAssign { out, .. } => out.precision().into_elem(),
            ElemwiseOp::Gather { output, .. } => output.precision().into_elem(),
            ElemwiseOp::Select { output, .. } => output.precision().into_elem(),
        }
    }
}

#[derive(CubeLaunch)]
pub struct ReshapedTensor {
    #[cube(comptime)]
    original: Arg,
    #[cube(comptime)]
    shape: Sequence<Arg>,
}

#[derive(CubeLaunch, Default)]
/// Global arguments that are used for fusing [element wise operations](ElemwiseOp).
pub struct GlobalArgs {
    pub tensors: Sequence<GlobalTensor>,
    pub scalars: Sequence<GlobalScalar>,
}

impl<R: Runtime> Default for GlobalArgsLaunch<'_, R> {
    fn default() -> Self {
        Self {
            tensors: Default::default(),
            scalars: Default::default(),
            _phantom_runtime: std::marker::PhantomData,
            _phantom_a: std::marker::PhantomData,
        }
    }
}

impl<R: Runtime> GlobalArgsLaunch<'_, R> {
    /// Get the shape of the given [argument](Arg).
    ///
    /// # Panics
    ///
    /// If the argument doesn't have an handle.
    pub fn shape(&self, arg: &Arg) -> &[usize] {
        match self.resolve_arg(arg) {
            TensorArg::Handle { handle, .. } => handle.shape,
            TensorArg::Alias { .. } => panic!("Unsupported yet"),
        }
    }

    /// Get the strides of the given [argument](Arg).
    ///
    /// # Panics
    ///
    /// If the argument doesn't have an handle.
    pub fn strides(&self, arg: &Arg) -> &[usize] {
        match self.resolve_arg(arg) {
            TensorArg::Handle { handle, .. } => handle.strides,
            TensorArg::Alias { .. } => panic!("Unsupported yet"),
        }
    }

    /// Get the line size of the given [argument](Arg).
    ///
    /// # Panics
    ///
    /// If the argument doesn't have an handle.
    pub fn line_size(&self, arg: &Arg) -> u8 {
        match self.resolve_arg(arg) {
            TensorArg::Handle {
                vectorization_factor,
                ..
            } => *vectorization_factor,
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
            _ => panic!("Only input & output can have a shape"),
        }
    }
}

#[derive(CubeType, Clone)]
/// Keep track of all local variables that are used as argument in fused
/// [element wise operations](ElemwiseOp).
pub struct LocalArgs {
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
}

#[cube]
impl LocalArgs {
    pub fn new() -> LocalArgs {
        LocalArgs {
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
        }
    }
}

#[derive(CubeType, Clone)]
/// Keep track of all local variables that are used as argument in fused
/// [element wise operations](ElemwiseOp).
pub struct LocalArgs2 {
    pub scalars: Registry<u32, Line<NumericExpand<DYN_ELEM_ID>>>,
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// Unary [element wise operation](ElemwiseOp) arguments.
pub struct UnaryElemwiseArgs {
    pub input: Arg,
    pub out: Arg,
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// Binary [element wise operation](ElemwiseOp) arguments.
pub struct BinaryElemwiseArgs {
    pub lhs: Arg,
    pub rhs: Arg,
    pub out: Arg,
}

#[derive(
    CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
/// Precisions supported by [element wise operations](ElemwiseOp).
pub enum ElemwisePrecision {
    F32,
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

impl From<Elem> for ElemwisePrecision {
    fn from(value: Elem) -> Self {
        match value {
            Elem::Float(kind) => match kind {
                cubecl::ir::FloatKind::F16 => Self::F16,
                cubecl::ir::FloatKind::BF16 => Self::BF16,
                cubecl::ir::FloatKind::F32 => Self::F32,
                _ => panic!("Unsupported precision for fusion: {value}"),
            },
            Elem::Int(kind) => match kind {
                cubecl::ir::IntKind::I64 => Self::I64,
                cubecl::ir::IntKind::I32 => Self::I32,
                cubecl::ir::IntKind::I16 => Self::I16,
                cubecl::ir::IntKind::I8 => Self::I8,
            },
            Elem::UInt(kind) => match kind {
                cubecl::ir::UIntKind::U64 => Self::U64,
                cubecl::ir::UIntKind::U32 => Self::U32,
                cubecl::ir::UIntKind::U16 => Self::U16,
                cubecl::ir::UIntKind::U8 => Self::U8,
            },
            Elem::Bool => Self::Bool,
            _ => panic!("Unsupported precision for fusion: {value}"),
        }
    }
}
impl ElemwisePrecision {
    pub fn into_elem(self) -> Elem {
        match self {
            ElemwisePrecision::F32 => Elem::Float(cubecl::ir::FloatKind::F32),
            ElemwisePrecision::F16 => Elem::Float(cubecl::ir::FloatKind::F16),
            ElemwisePrecision::BF16 => Elem::Float(cubecl::ir::FloatKind::BF16),
            ElemwisePrecision::I64 => Elem::Int(cubecl::ir::IntKind::I64),
            ElemwisePrecision::I32 => Elem::Int(cubecl::ir::IntKind::I32),
            ElemwisePrecision::I16 => Elem::Int(cubecl::ir::IntKind::I16),
            ElemwisePrecision::I8 => Elem::Int(cubecl::ir::IntKind::I8),
            ElemwisePrecision::U64 => Elem::UInt(cubecl::ir::UIntKind::U64),
            ElemwisePrecision::U32 => Elem::UInt(cubecl::ir::UIntKind::U32),
            ElemwisePrecision::U16 => Elem::UInt(cubecl::ir::UIntKind::U16),
            ElemwisePrecision::U8 => Elem::UInt(cubecl::ir::UIntKind::U8),
            ElemwisePrecision::Bool => Elem::Bool,
        }
    }
}

impl From<DType> for ElemwisePrecision {
    fn from(value: DType) -> Self {
        match value {
            DType::F32 => Self::F32,
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
            _ => panic!("Unsupported"),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// Configuration that encapsulates all comptime information necessary for element wise fusion.
pub struct ElemwiseConfig {
    pub rank: u32,
    pub ref_layout: Arg,
    pub ops: Sequence<ElemwiseOp>,
    pub width: u8,
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
