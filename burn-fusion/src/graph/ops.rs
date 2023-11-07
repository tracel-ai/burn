use crate::FusedBackend;
use crate::{HandleContainer, TensorDescription};
use burn_tensor::{
    ops::{ConvOptions, ConvTransposeOptions},
    Distribution, Element,
};
use std::ops::Range;

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
            TensorOps::NumericOpsFloat(ops) => ops.cleanup_tensor(handles),
            TensorOps::NumericOpsInt(ops) => ops.cleanup_tensor(handles),
            TensorOps::BoolOps(_) => todo!(),
            TensorOps::IntOps(_) => todo!(),
            TensorOps::FloatOps(_) => todo!(),
            TensorOps::ModuleOps(_) => todo!(),
        }
    }
    pub(crate) fn execute(&self, handles: &mut HandleContainer<B>) {
        match self {
            TensorOps::BaseOpsFloat(ops) => ops.execute(handles),
            TensorOps::BaseOpsInt(ops) => ops.execute(handles),
            TensorOps::BaseOpsBool(_) => todo!(),
            TensorOps::NumericOpsFloat(ops) => ops.execute(handles),
            TensorOps::NumericOpsInt(ops) => ops.execute(handles),
            TensorOps::BoolOps(_) => todo!(),
            TensorOps::IntOps(_) => todo!(),
            TensorOps::FloatOps(_) => todo!(),
            TensorOps::ModuleOps(_) => todo!(),
        }
    }
}

impl<B: FusedBackend, E> BaseOps<B, E> {
    fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        todo!();
    }
    fn execute(&self, handles: &mut HandleContainer<B>) {
        todo!();
    }
}

impl<B: FusedBackend, E: Element> NumericOps<B, E> {
    fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        match self {
            NumericOps::Add(desc, ops) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOps::AddScalar(desc, ops) => {
                handles.cleanup(&desc.lhs);
            }
            _ => todo!(),
        }
    }
    fn execute(&self, handles: &mut HandleContainer<B>) {
        match self {
            NumericOps::Add(desc, ops) => ops.execute(desc, handles),
            NumericOps::AddScalar(desc, ops) => ops.execute(desc, handles),
            _ => todo!(),
        }
    }
}

pub enum FloatOps<B: FusedBackend, E: core::fmt::Debug> {
    Exp {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    Log {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    Log1p {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    Erf {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    Powf {
        tensor: TensorDescription,
        value: E,
        out: TensorDescription,
    },
    Sqrt {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    Cos {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    Sin {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    Tanh {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    IntoInt {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    Matmul {
        lhs: TensorDescription,
        rhs: TensorDescription,
        out: TensorDescription,
    },
    Random {
        distribution: Distribution<E>,
        out: TensorDescription,
        ops: Box<dyn Ops<B, Args = (TensorDescription, Distribution<E>)>>,
    },
}

#[derive(Debug)]
pub enum ModuleOps {
    Embedding {
        weights: TensorDescription,
        indices: TensorDescription,
        out: TensorDescription,
    },
    EmbeddingBackward {
        weights: TensorDescription,
        out_grad: TensorDescription,
        indices: TensorDescription,
        out: TensorDescription,
    },
    Conv1d {
        x: TensorDescription,
        weight: TensorDescription,
        bias: Option<TensorDescription>,
        options: ConvOptions<1>,
        out: TensorDescription,
    },
    Conv2d {
        x: TensorDescription,
        weight: TensorDescription,
        bias: Option<TensorDescription>,
        options: ConvOptions<2>,
        out: TensorDescription,
    },
    ConvTranspose1d {
        x: TensorDescription,
        weight: TensorDescription,
        bias: Option<TensorDescription>,
        options: ConvTransposeOptions<1>,
        out: TensorDescription,
    },
    ConvTranspose2d {
        x: TensorDescription,
        weight: TensorDescription,
        bias: Option<TensorDescription>,
        options: ConvTransposeOptions<2>,
        out: TensorDescription,
    },
    AvgPool1d {
        x: TensorDescription,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        out: TensorDescription,
    },
    AvgPool2d {
        x: TensorDescription,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        out: TensorDescription,
    },
    AvgPool1dBackward {
        x: TensorDescription,
        grad: TensorDescription,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        out: TensorDescription,
    },
    AvgPool2dBackward {
        x: TensorDescription,
        grad: TensorDescription,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        out: TensorDescription,
    },
    AdaptiveAvgPool1d {
        x: TensorDescription,
        output_size: usize,
        out: TensorDescription,
    },
    AdaptiveAvgPool2d {
        x: TensorDescription,
        output_size: [usize; 2],
        out: TensorDescription,
    },
    AdaptiveAvgPool1dBackward {
        x: TensorDescription,
        grad: TensorDescription,
        out: TensorDescription,
    },
    AdaptiveAvgPool2dBackward {
        x: TensorDescription,
        grad: TensorDescription,
        out: TensorDescription,
    },
    MaxPool1d {
        x: TensorDescription,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        out: TensorDescription,
    },
    MaxPool1dWithIndices {
        x: TensorDescription,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        out: TensorDescription,
        out_indices: TensorDescription,
    },
    MaxPool1dWithIndicesBackward {
        x: TensorDescription,
        grad: TensorDescription,
        indices: TensorDescription,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        out: TensorDescription,
    },
    MaxPool2d {
        x: TensorDescription,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        out: TensorDescription,
    },
    MaxPool2dWithIndices {
        x: TensorDescription,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        out: TensorDescription,
        out_indices: TensorDescription,
    },
    MaxPool2dWithIndicesBackward {
        x: TensorDescription,
        grad: TensorDescription,
        indices: TensorDescription,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        out: TensorDescription,
    },
}

pub enum BaseOps<B: FusedBackend, E> {
    Reshape {
        tensor: TensorDescription,
        shape: Vec<usize>,
        out: TensorDescription,
        ops: Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    },
    SwapDims {
        tensor: TensorDescription,
        dim1: usize,
        dim2: usize,
        out: TensorDescription,
    },
    Slice {
        tensor: TensorDescription,
        ranges: Vec<Range<usize>>,
        out: TensorDescription,
    },
    SliceAssign {
        tensor: TensorDescription,
        ranges: Vec<Range<usize>>,
        values: TensorDescription,
        out: TensorDescription,
    },
    FromData {
        value: Vec<E>,
        shape: Vec<usize>,
        out: TensorDescription,
    },
    Repeat {
        tensor: TensorDescription,
        dim: usize,
        times: usize,
        shape: Vec<usize>,
        out: TensorDescription,
    },
    Equal {
        lhs: TensorDescription,
        rhs: TensorDescription,
        out: TensorDescription,
    },
    Cat {
        tensors: Vec<TensorDescription>,
        dim: usize,
        out: TensorDescription,
    },
}

pub trait Ops<B: FusedBackend>: Send + Sync {
    type Args: Send + Sync;

    fn execute(&self, args: &Self::Args, handles: &mut HandleContainer<B>);
}

pub struct BinaryOpsDescription {
    pub lhs: TensorDescription,
    pub rhs: TensorDescription,
    pub out: TensorDescription,
}

pub struct ScalarOpsDescription<E> {
    pub lhs: TensorDescription,
    pub rhs: E,
    pub out: TensorDescription,
}

pub enum NumericOps<B: FusedBackend, E: Element> {
    Add(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    AddScalar(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    Sub {
        lhs: TensorDescription,
        rhs: TensorDescription,
        out: TensorDescription,
    },
    SubScalar {
        lhs: TensorDescription,
        rhs: E,
        out: TensorDescription,
    },
    Div {
        lhs: TensorDescription,
        rhs: TensorDescription,
        out: TensorDescription,
    },
    DivScalar {
        lhs: TensorDescription,
        rhs: E,
        out: TensorDescription,
    },
    Mul {
        lhs: TensorDescription,
        rhs: TensorDescription,
        out: TensorDescription,
    },
    MulScalar {
        lhs: TensorDescription,
        rhs: E,
        out: TensorDescription,
    },
    Neg {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    Abs {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    Zeros {
        shape: Vec<usize>,
        out: TensorDescription,
    },
    Ones {
        shape: Vec<usize>,
        out: TensorDescription,
    },
    Full {
        shape: Vec<usize>,
        value: E,
        out: TensorDescription,
    },
    Mean {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    MeanDim {
        tensor: TensorDescription,
        dim: usize,
        out: TensorDescription,
    },
    Sum {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    SumDim {
        tensor: TensorDescription,
        dim: usize,
        out: TensorDescription,
    },
    EqualElem {
        tensor: TensorDescription,
        elem: E,
        out: TensorDescription,
    },
    Greater {
        lhs: TensorDescription,
        rhs: TensorDescription,
        out: TensorDescription,
    },
    GreaterElem {
        tensor: TensorDescription,
        elem: E,
        out: TensorDescription,
    },
    GreaterEqual {
        lhs: TensorDescription,
        rhs: TensorDescription,
        out: TensorDescription,
    },
    GreaterEqualElem {
        tensor: TensorDescription,
        elem: E,
        out: TensorDescription,
    },
    Lower {
        lhs: TensorDescription,
        rhs: TensorDescription,
        out: TensorDescription,
    },
    LowerElem {
        tensor: TensorDescription,
        elem: E,
        out: TensorDescription,
    },
    LowerEqual {
        lhs: TensorDescription,
        rhs: TensorDescription,
        out: TensorDescription,
    },
    LowerEqualElem {
        tensor: TensorDescription,
        elem: E,
        out: TensorDescription,
    },
    MaskWhere {
        tensor: TensorDescription,
        mask: TensorDescription,
        value: TensorDescription,
        out: TensorDescription,
    },
    MaskFill {
        tensor: TensorDescription,
        mask: TensorDescription,
        value: E,
        out: TensorDescription,
    },
    Gather {
        tensor: TensorDescription,
        dim: usize,
        indices: TensorDescription,
        out: TensorDescription,
    },
    Scatter {
        tensor: TensorDescription,
        dim: usize,
        indices: TensorDescription,
        values: TensorDescription,
        out: TensorDescription,
    },
    Select {
        tensor: TensorDescription,
        dim: usize,
        indices: TensorDescription,
    },
    SelectAssign {
        tensor: TensorDescription,
        dim: usize,
        indices: TensorDescription,
        values: TensorDescription,
    },
    ArgMax {
        tensor: TensorDescription,
        dim: usize,
        out: TensorDescription,
    },
    ArgMin {
        tensor: TensorDescription,
        dim: usize,
        out: TensorDescription,
    },
    Max {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    MaxDim {
        tensor: TensorDescription,
        dim: usize,
        out: TensorDescription,
    },
    MaxDimWithIndices {
        tensor: TensorDescription,
        dim: usize,
        out: TensorDescription,
        out_indices: TensorDescription,
    },
    Min {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    MinDim {
        tensor: TensorDescription,
        dim: usize,
        out: TensorDescription,
    },
    Clamp {
        tensor: TensorDescription,
        min: usize,
        max: usize,
        out: TensorDescription,
    },
    ClampMax {
        tensor: TensorDescription,
        max: usize,
        out: TensorDescription,
    },
    ClampMin {
        tensor: TensorDescription,
        min: usize,
        out: TensorDescription,
    },
}

#[derive(Debug)]
pub enum IntOps {
    Arange {
        tensor: TensorDescription,
        range: Range<usize>,
        out: TensorDescription,
    },
    ArangeStep {
        tensor: TensorDescription,
        range: Range<usize>,
        step: usize,
        out: TensorDescription,
    },
    IntoFloat {
        tensor: TensorDescription,
        out: TensorDescription,
    },
}

#[derive(Debug)]
pub enum BoolOps {
    IntoFloat {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    IntoInt {
        tensor: TensorDescription,
        out: TensorDescription,
    },
    Not {
        tensor: TensorDescription,
        out: TensorDescription,
    },
}
