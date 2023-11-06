use crate::{HandleContainer, TensorDefinition};
use burn_tensor::{
    ops::{ConvOptions, ConvTransposeOptions},
    Distribution, Element,
};
use std::ops::Range;

use super::FusedBackend;

pub enum TensorOps<B: FusedBackend> {
    BaseOpsFloat(BaseOps<B, B::FloatElem>),
    BaseOpsInt(BaseOps<B, B::IntElem>),
    BaseOpsBool(BaseOps<B, bool>),
    NumericOpsFloat(NumericOps<B, B::FloatElem>),
    NumericOpsInt(NumericOps<B, B::IntElem>),
    BoolOps(BoolOps),
    IntOps(IntOps),
    FloatOps(FloatOps<B, B::FloatElem>),
    ModuleOps(ModuleOps),
}

impl<B: FusedBackend> TensorOps<B> {
    pub(crate) fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        match self {
            TensorOps::BaseOpsFloat(ops) => ops.cleanup_tensor(handles),
            TensorOps::BaseOpsInt(ops) => ops.cleanup_tensor(handles),
            TensorOps::BaseOpsBool(_) => todo!(),
            TensorOps::NumericOpsFloat(_) => todo!(),
            TensorOps::NumericOpsInt(_) => todo!(),
            TensorOps::BoolOps(_) => todo!(),
            TensorOps::IntOps(_) => todo!(),
            TensorOps::FloatOps(_) => todo!(),
            TensorOps::ModuleOps(_) => todo!(),
        }
    }
}
impl<B: FusedBackend, E> BaseOps<B, E> {
    fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        match self {
            BaseOps::Reshape {
                tensor,
                shape,
                out,
                ops,
            } => handles.cleanup(&tensor),
            BaseOps::SwapDims {
                tensor,
                dim1,
                dim2,
                out,
            } => todo!(),
            BaseOps::Slice {
                tensor,
                ranges,
                out,
            } => todo!(),
            BaseOps::SliceAssign {
                tensor,
                ranges,
                values,
                out,
            } => todo!(),
            BaseOps::FromData { value, shape, out } => todo!(),
            BaseOps::Repeat {
                tensor,
                dim,
                times,
                shape,
                out,
            } => todo!(),
            BaseOps::Equal { lhs, rhs, out } => todo!(),
            BaseOps::Cat { tensors, dim, out } => todo!(),
        }
    }
}

#[derive(Debug)]
pub enum FloatOps<B: FusedBackend, E: core::fmt::Debug> {
    Exp {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    Log {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    Log1p {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    Erf {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    Powf {
        tensor: TensorDefinition,
        value: E,
        out: TensorDefinition,
    },
    Sqrt {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    Cos {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    Sin {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    Tanh {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    IntoInt {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    Matmul {
        lhs: TensorDefinition,
        rhs: TensorDefinition,
        out: TensorDefinition,
    },
    Random {
        distribution: Distribution<E>,
        out: TensorDefinition,
        ops: Box<dyn Ops<B, Args = (TensorDefinition, Distribution<E>)>>,
    },
}

#[derive(Debug)]
pub enum ModuleOps {
    Embedding {
        weights: TensorDefinition,
        indices: TensorDefinition,
        out: TensorDefinition,
    },
    EmbeddingBackward {
        weights: TensorDefinition,
        out_grad: TensorDefinition,
        indices: TensorDefinition,
        out: TensorDefinition,
    },
    Conv1d {
        x: TensorDefinition,
        weight: TensorDefinition,
        bias: Option<TensorDefinition>,
        options: ConvOptions<1>,
        out: TensorDefinition,
    },
    Conv2d {
        x: TensorDefinition,
        weight: TensorDefinition,
        bias: Option<TensorDefinition>,
        options: ConvOptions<2>,
        out: TensorDefinition,
    },
    ConvTranspose1d {
        x: TensorDefinition,
        weight: TensorDefinition,
        bias: Option<TensorDefinition>,
        options: ConvTransposeOptions<1>,
        out: TensorDefinition,
    },
    ConvTranspose2d {
        x: TensorDefinition,
        weight: TensorDefinition,
        bias: Option<TensorDefinition>,
        options: ConvTransposeOptions<2>,
        out: TensorDefinition,
    },
    AvgPool1d {
        x: TensorDefinition,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        out: TensorDefinition,
    },
    AvgPool2d {
        x: TensorDefinition,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        out: TensorDefinition,
    },
    AvgPool1dBackward {
        x: TensorDefinition,
        grad: TensorDefinition,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        out: TensorDefinition,
    },
    AvgPool2dBackward {
        x: TensorDefinition,
        grad: TensorDefinition,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        out: TensorDefinition,
    },
    AdaptiveAvgPool1d {
        x: TensorDefinition,
        output_size: usize,
        out: TensorDefinition,
    },
    AdaptiveAvgPool2d {
        x: TensorDefinition,
        output_size: [usize; 2],
        out: TensorDefinition,
    },
    AdaptiveAvgPool1dBackward {
        x: TensorDefinition,
        grad: TensorDefinition,
        out: TensorDefinition,
    },
    AdaptiveAvgPool2dBackward {
        x: TensorDefinition,
        grad: TensorDefinition,
        out: TensorDefinition,
    },
    MaxPool1d {
        x: TensorDefinition,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        out: TensorDefinition,
    },
    MaxPool1dWithIndices {
        x: TensorDefinition,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        out: TensorDefinition,
        out_indices: TensorDefinition,
    },
    MaxPool1dWithIndicesBackward {
        x: TensorDefinition,
        grad: TensorDefinition,
        indices: TensorDefinition,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        out: TensorDefinition,
    },
    MaxPool2d {
        x: TensorDefinition,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        out: TensorDefinition,
    },
    MaxPool2dWithIndices {
        x: TensorDefinition,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        out: TensorDefinition,
        out_indices: TensorDefinition,
    },
    MaxPool2dWithIndicesBackward {
        x: TensorDefinition,
        grad: TensorDefinition,
        indices: TensorDefinition,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        out: TensorDefinition,
    },
}

pub enum BaseOps<B: FusedBackend, E> {
    Reshape {
        tensor: TensorDefinition,
        shape: Vec<usize>,
        out: TensorDefinition,
        ops: Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    },
    SwapDims {
        tensor: TensorDefinition,
        dim1: usize,
        dim2: usize,
        out: TensorDefinition,
    },
    Slice {
        tensor: TensorDefinition,
        ranges: Vec<Range<usize>>,
        out: TensorDefinition,
    },
    SliceAssign {
        tensor: TensorDefinition,
        ranges: Vec<Range<usize>>,
        values: TensorDefinition,
        out: TensorDefinition,
    },
    FromData {
        value: Vec<E>,
        shape: Vec<usize>,
        out: TensorDefinition,
    },
    Repeat {
        tensor: TensorDefinition,
        dim: usize,
        times: usize,
        shape: Vec<usize>,
        out: TensorDefinition,
    },
    Equal {
        lhs: TensorDefinition,
        rhs: TensorDefinition,
        out: TensorDefinition,
    },
    Cat {
        tensors: Vec<TensorDefinition>,
        dim: usize,
        out: TensorDefinition,
    },
}

pub trait Ops<B: FusedBackend>: Send + Sync {
    type Args: Send + Sync;

    fn execute(self: Box<Self>, args: Self::Args, handles: &mut HandleContainer<B>);
}

pub struct BinaryOpsDescription {
    pub lhs: TensorDefinition,
    pub rhs: TensorDefinition,
    pub out: TensorDefinition,
}

pub enum NumericOps<B: FusedBackend, E: Element> {
    Add(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    AddScalar {
        lhs: TensorDefinition,
        rhs: E,
        out: TensorDefinition,
    },
    Sub {
        lhs: TensorDefinition,
        rhs: TensorDefinition,
        out: TensorDefinition,
    },
    SubScalar {
        lhs: TensorDefinition,
        rhs: E,
        out: TensorDefinition,
    },
    Div {
        lhs: TensorDefinition,
        rhs: TensorDefinition,
        out: TensorDefinition,
    },
    DivScalar {
        lhs: TensorDefinition,
        rhs: E,
        out: TensorDefinition,
    },
    Mul {
        lhs: TensorDefinition,
        rhs: TensorDefinition,
        out: TensorDefinition,
    },
    MulScalar {
        lhs: TensorDefinition,
        rhs: E,
        out: TensorDefinition,
    },
    Neg {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    Abs {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    Zeros {
        shape: Vec<usize>,
        out: TensorDefinition,
    },
    Ones {
        shape: Vec<usize>,
        out: TensorDefinition,
    },
    Full {
        shape: Vec<usize>,
        value: E,
        out: TensorDefinition,
    },
    Mean {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    MeanDim {
        tensor: TensorDefinition,
        dim: usize,
        out: TensorDefinition,
    },
    Sum {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    SumDim {
        tensor: TensorDefinition,
        dim: usize,
        out: TensorDefinition,
    },
    EqualElem {
        tensor: TensorDefinition,
        elem: E,
        out: TensorDefinition,
    },
    Greater {
        lhs: TensorDefinition,
        rhs: TensorDefinition,
        out: TensorDefinition,
    },
    GreaterElem {
        tensor: TensorDefinition,
        elem: E,
        out: TensorDefinition,
    },
    GreaterEqual {
        lhs: TensorDefinition,
        rhs: TensorDefinition,
        out: TensorDefinition,
    },
    GreaterEqualElem {
        tensor: TensorDefinition,
        elem: E,
        out: TensorDefinition,
    },
    Lower {
        lhs: TensorDefinition,
        rhs: TensorDefinition,
        out: TensorDefinition,
    },
    LowerElem {
        tensor: TensorDefinition,
        elem: E,
        out: TensorDefinition,
    },
    LowerEqual {
        lhs: TensorDefinition,
        rhs: TensorDefinition,
        out: TensorDefinition,
    },
    LowerEqualElem {
        tensor: TensorDefinition,
        elem: E,
        out: TensorDefinition,
    },
    MaskWhere {
        tensor: TensorDefinition,
        mask: TensorDefinition,
        value: TensorDefinition,
        out: TensorDefinition,
    },
    MaskFill {
        tensor: TensorDefinition,
        mask: TensorDefinition,
        value: E,
        out: TensorDefinition,
    },
    Gather {
        tensor: TensorDefinition,
        dim: usize,
        indices: TensorDefinition,
        out: TensorDefinition,
    },
    Scatter {
        tensor: TensorDefinition,
        dim: usize,
        indices: TensorDefinition,
        values: TensorDefinition,
        out: TensorDefinition,
    },
    Select {
        tensor: TensorDefinition,
        dim: usize,
        indices: TensorDefinition,
    },
    SelectAssign {
        tensor: TensorDefinition,
        dim: usize,
        indices: TensorDefinition,
        values: TensorDefinition,
    },
    ArgMax {
        tensor: TensorDefinition,
        dim: usize,
        out: TensorDefinition,
    },
    ArgMin {
        tensor: TensorDefinition,
        dim: usize,
        out: TensorDefinition,
    },
    Max {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    MaxDim {
        tensor: TensorDefinition,
        dim: usize,
        out: TensorDefinition,
    },
    MaxDimWithIndices {
        tensor: TensorDefinition,
        dim: usize,
        out: TensorDefinition,
        out_indices: TensorDefinition,
    },
    Min {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    MinDim {
        tensor: TensorDefinition,
        dim: usize,
        out: TensorDefinition,
    },
    Clamp {
        tensor: TensorDefinition,
        min: usize,
        max: usize,
        out: TensorDefinition,
    },
    ClampMax {
        tensor: TensorDefinition,
        max: usize,
        out: TensorDefinition,
    },
    ClampMin {
        tensor: TensorDefinition,
        min: usize,
        out: TensorDefinition,
    },
}

#[derive(Debug)]
pub enum IntOps {
    Arange {
        tensor: TensorDefinition,
        range: Range<usize>,
        out: TensorDefinition,
    },
    ArangeStep {
        tensor: TensorDefinition,
        range: Range<usize>,
        step: usize,
        out: TensorDefinition,
    },
    IntoFloat {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
}

#[derive(Debug)]
pub enum BoolOps {
    IntoFloat {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    IntoInt {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
    Not {
        tensor: TensorDefinition,
        out: TensorDefinition,
    },
}
