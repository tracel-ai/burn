use burn_std::DType;
use burn_std::quantization::{QuantScheme, QuantStore, QuantValue};
use burn_std::{bf16, f16};
use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, UIntKind};
use cubecl::prelude::*;
use serde::{Deserialize, Serialize};

use crate::engine::codegen::DYN_ELEM_ID;
use crate::engine::trace::block::LocalInput;

use super::tensor::GlobalTensor;

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
/// Argument to a [fuse operation](FuseOp).
pub enum FuseArg {
    /// A readonly input tensor.
    Input(usize, FuseType, LayoutInfo),
    /// A temporary local variable.
    Local(usize, FuseType),
    /// A permanent register shared between blocks.
    GlobalRegister((usize, usize), FuseType),
    /// A readwrite output tensor.
    Output(usize, FuseType, LayoutInfo),
    /// A global scalar.
    Scalar(usize, FuseType),
    /// A global scalar used in a reshape operation.
    ///
    /// This is not a scalar defined by a user for computation, but a scalar defined as part of
    /// a reshape operation.
    ScalarShape(usize),
    /// Only constant that can be encoded into an u32 can be used as literal.
    Literal(usize, FuseType),
    /// A readonly input tensor that is reshaped.
    InputReshaped {
        original: Box<FuseArg>,
        shape: Vec<FuseArg>,
        broadcasted: bool,
    },
    /// A readonly input tensor with swapped dimensions.
    InputSwapDims {
        original: Box<FuseArg>,
        dims: (usize, usize),
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

impl FuseArg {
    pub fn precision(&self) -> FuseType {
        *match self {
            FuseArg::Input(_, p, _) => p,
            FuseArg::Local(_, p) => p,
            FuseArg::GlobalRegister(_, p) => p,
            FuseArg::Output(_, p, _) => p,
            FuseArg::Scalar(_, p) => p,
            FuseArg::Literal(_, p) => p,
            FuseArg::ScalarShape(_) => return FuseType::U32,
            FuseArg::InputReshaped { original, .. } => return original.precision(),
            FuseArg::InputSwapDims { original, .. } => return original.precision(),
        }
    }
}

impl CubeType for FuseArg {
    type ExpandType = Self;
}

impl IntoMut for FuseArg {
    fn into_mut(self, _context: &mut Scope) -> Self {
        self
    }
}

impl IntoRuntime for FuseArg {
    fn __expand_runtime_method(self, _context: &mut Scope) -> Self::ExpandType {
        self
    }
}

impl CubeDebug for FuseArg {}

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
        input: FuseArg,
        min: FuseArg,
        max: FuseArg,
        out: FuseArg,
    },
    ConditionalAssign {
        cond: FuseArg,
        lhs: FuseArg,
        rhs: FuseArg,
        out: FuseArg,
    },
    Gather {
        input: FuseArg,
        indices: FuseArg,
        output: FuseArg,
        dim: usize,
    },
    Select {
        input: FuseArg,
        indices: FuseArg,
        output: FuseArg,
        dim: usize,
    },
    Dequantize {
        values: FuseArg,
        params: FuseArg,
        output: FuseArg,
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
    pub scalars: Sequence<InputScalar>,
    pub reshapes: Sequence<usize>,
    pub registers: GlobalRegisters,
}

#[derive(CubeType, Default, Clone)]
pub struct GlobalRegisters {
    registers: Registry<usize, Registry<usize, Line<NumericExpand<DYN_ELEM_ID>>>>,
}

#[cube]
impl GlobalRegisters {
    pub fn read(&self, #[comptime] key: (usize, usize)) -> Line<NumericExpand<DYN_ELEM_ID>> {
        let registers = self.registers.find(key.0);
        registers.find(key.1)
    }
    pub fn write(
        &mut self,
        #[comptime] key: (usize, usize),
        value: Line<NumericExpand<DYN_ELEM_ID>>,
    ) {
        // TODO: Implement try find.
        let mut registers =
            Registry::<usize, Registry<usize, Line<NumericExpand<DYN_ELEM_ID>>>>::find_or_default::<
                usize,
            >(&mut self.registers, key.0);
        registers.insert(key.1, value);
    }
}

// Because we only create it DURING compilation, not as a real launch arg.
unsafe impl Send for GlobalRegisters {}
unsafe impl Sync for GlobalRegisters {}

impl LaunchArg for GlobalRegisters {
    type RuntimeArg<'a, R: Runtime> = ();
    type CompilationArg = ();

    fn compilation_arg<R: Runtime>(_runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        ()
    }

    fn expand(
        _arg: &Self::CompilationArg,
        _builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        GlobalRegistersExpand {
            registers: Default::default(),
        }
    }
}

impl<R: Runtime> Default for GlobalArgsLaunch<'_, R> {
    fn default() -> Self {
        Self {
            tensors: Default::default(),
            scalars: Default::default(),
            reshapes: Default::default(),
            registers: Default::default(),
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
    pub fn shape(&self, arg: &FuseArg) -> Vec<usize> {
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
                    shape.swap(dims.0, dims.1);
                    shape
                }
                VirtualLayout::Reshaped { reshape_pos, .. } => {
                    let start = *reshape_pos * rank;
                    let end = start + rank;
                    self.reshapes.values[start..end]
                        .iter()
                        .map(|s| s.elem)
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
    pub fn strides(&self, arg: &FuseArg) -> Vec<usize> {
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
    pub fn line_size(&self, arg: &FuseArg) -> LineSize {
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
    pub fn resolve_arg(&self, arg: &FuseArg) -> &TensorArg<'_, R> {
        match arg {
            FuseArg::Input(pos, _, _) => &self.tensors.values[*pos].tensor,
            FuseArg::Output(pos, _, _) => &self.tensors.values[*pos].tensor,
            other => panic!("Arg not found: {other:?}"),
        }
    }
}

#[derive(CubeType, Clone)]
/// Keep track of all local variables that are used as argument in fused
/// [element wise operations](ElemwiseOp).
pub struct LocalArgs {
    pub l_f64: Registry<usize, Line<f64>>,
    pub l_f32: Registry<usize, Line<f32>>,
    pub l_f16: Registry<usize, Line<f16>>,
    pub l_bf16: Registry<usize, Line<bf16>>,
    pub l_i64: Registry<usize, Line<i64>>,
    pub l_i32: Registry<usize, Line<i32>>,
    pub l_i16: Registry<usize, Line<i16>>,
    pub l_i8: Registry<usize, Line<i8>>,
    pub l_u64: Registry<usize, Line<u64>>,
    pub l_u32: Registry<usize, Line<u32>>,
    pub l_u16: Registry<usize, Line<u16>>,
    pub l_u8: Registry<usize, Line<u8>>,
    pub l_bool: Registry<usize, Line<bool>>,
    pub ref_shape: Slice<usize>,
    pub ref_strides: Slice<usize>,
    #[cube(comptime)]
    pub ref_line_size: LineSize,
}

#[cube]
impl LocalArgs {
    /// Creates a new [LocalArgs] container.
    pub fn new(
        ref_shape: Slice<usize>,
        ref_strides: Slice<usize>,
        #[comptime] ref_line_size: LineSize,
    ) -> LocalArgs {
        LocalArgs {
            l_f64: Registry::<usize, Line<f64>>::new(),
            l_f32: Registry::<usize, Line<f32>>::new(),
            l_f16: Registry::<usize, Line<f16>>::new(),
            l_bf16: Registry::<usize, Line<bf16>>::new(),
            l_i64: Registry::<usize, Line<i64>>::new(),
            l_i32: Registry::<usize, Line<i32>>::new(),
            l_i16: Registry::<usize, Line<i16>>::new(),
            l_i8: Registry::<usize, Line<i8>>::new(),
            l_u64: Registry::<usize, Line<u64>>::new(),
            l_u32: Registry::<usize, Line<u32>>::new(),
            l_u16: Registry::<usize, Line<u16>>::new(),
            l_u8: Registry::<usize, Line<u8>>::new(),
            l_bool: Registry::<usize, Line<bool>>::new(),
            ref_shape,
            ref_strides,
            ref_line_size,
        }
    }
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// Unary [element wise operation](ElemwiseOp) arguments.
pub struct UnaryFuseArgs {
    pub input: FuseArg,
    pub out: FuseArg,
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// Binary [element wise operation](ElemwiseOp) arguments.
pub struct BinaryFuseArgs {
    pub lhs: FuseArg,
    pub rhs: FuseArg,
    pub out: FuseArg,
}

#[derive(
    CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize,
)]
/// Precisions supported by [element wise operations](ElemwiseOp).
///
/// This is a custom type instead of [ElemType] so it can implement [CubeType]
/// and restricts the supported types for fusion.
pub enum FuseType {
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

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// Configuration that encapsulates all comptime information necessary for element wise fusion.
pub struct FuseBlockConfig {
    pub rank: usize,
    pub ref_layout: RefLayout,
    pub ops: Vec<FuseOp>,
    pub width: LineSize,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// A reference layout determines how a fuse execution will access elements in tensors.
///
/// It can either follow the same layout as a concrete tensor, or follow a virtual layout.
pub enum RefLayout {
    Concrete(FuseArg),
    Virtual(VirtualLayout),
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
/// A virtual layout is always contiguous and retrieves its shape from either a reshaped tensor or a
/// tensor with swap dimensions.
pub enum VirtualLayout {
    /// Virtual tensor with the provided shape id and contiguous strides.
    Reshaped {
        reshape_pos: usize,
        line_size: LineSize,
    },
    /// Virtual tensor with the same shape as the given input, but with swap dims and contiguous
    /// strides.
    SwapDims(FuseArg, (usize, usize)),
    /// Virtual tensor with the same shape as the given input, but with contiguous strides.
    Shape(FuseArg, usize),
}

impl FuseArg {
    /// Adds layout information.
    ///
    /// It's going to impact how the input or output is read and written to.
    pub fn add_layout_info(&mut self, layout: LayoutInfo) {
        match self {
            FuseArg::Input(_, _, old) => {
                *old = layout;
            }
            FuseArg::Output(_, _, old) => {
                *old = layout;
            }
            _ => {}
        }
    }
}

impl RegistryQuery<Self> for FuseArg {}

impl From<ElemType> for FuseType {
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

impl From<StorageType> for FuseType {
    fn from(value: StorageType) -> Self {
        value.elem_type().into()
    }
}

impl FuseType {
    /// Converts the [fused element type](FuseType) into the [cubecl element type](ElemType).
    pub fn into_elem(self) -> ElemType {
        match self {
            FuseType::F32 => ElemType::Float(FloatKind::F32),
            FuseType::Flex32 => ElemType::Float(FloatKind::Flex32),
            FuseType::F16 => ElemType::Float(FloatKind::F16),
            FuseType::BF16 => ElemType::Float(FloatKind::BF16),
            FuseType::I64 => ElemType::Int(IntKind::I64),
            FuseType::I32 => ElemType::Int(IntKind::I32),
            FuseType::I16 => ElemType::Int(IntKind::I16),
            FuseType::I8 => ElemType::Int(IntKind::I8),
            FuseType::U64 => ElemType::UInt(UIntKind::U64),
            FuseType::U32 => ElemType::UInt(UIntKind::U32),
            FuseType::U16 => ElemType::UInt(UIntKind::U16),
            FuseType::U8 => ElemType::UInt(UIntKind::U8),
            FuseType::Bool => ElemType::Bool,
            FuseType::F64 => ElemType::Float(FloatKind::F64),
        }
    }

    /// Convert the [fused element type](FuseType) into the [cubecl storage type](StorageType).
    pub fn into_type(self) -> StorageType {
        self.into_elem().into()
    }
}

impl From<DType> for FuseType {
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
                    QuantValue::E4M3 | QuantValue::E5M2 => {
                        unimplemented!("Unsupported precision for fusion")
                    }
                    QuantValue::Q4F
                    | QuantValue::Q4S
                    | QuantValue::Q2F
                    | QuantValue::Q2S
                    | QuantValue::E2M1 => {
                        panic!("Can't store native sub-byte values")
                    }
                },
                QuantStore::PackedU32(_) => Self::U32,
                QuantStore::PackedNative(_) => match scheme.value {
                    QuantValue::E2M1 => unimplemented!("Unsupported precision for fusion"),
                    other => panic!("{other:?} doesn't support native packing"),
                },
            },
        }
    }
}
