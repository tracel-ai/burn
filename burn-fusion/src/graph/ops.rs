use crate::FusedBackend;
use crate::{HandleContainer, TensorDescription};
use burn_tensor::ops::FloatElem;
use burn_tensor::{
    ops::{ConvOptions, ConvTransposeOptions},
    Distribution, Element,
};
use std::ops::Range;

pub enum TensorOpsDescription<B: FusedBackend> {
    BaseOpsFloat(BaseOpsDescription<B>),
    BaseOpsInt(BaseOpsDescription<B>),
    BaseOpsBool(BaseOpsDescription<B>),
    NumericOpsFloat(NumericOpsDescription<B, B::FloatElem>),
    NumericOpsInt(NumericOpsDescription<B, B::IntElem>),
    BoolOps(BoolOpsDescription<B>),
    IntOps(IntOpsDescription<B>),
    FloatOps(FloatOpsDescription<B>),
    ModuleOps(ModuleOpsDescription),
}

impl<B: FusedBackend> TensorOpsDescription<B> {
    pub(crate) fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        match self {
            TensorOpsDescription::BaseOpsFloat(ops) => ops.cleanup_tensor(handles),
            TensorOpsDescription::BaseOpsInt(ops) => ops.cleanup_tensor(handles),
            TensorOpsDescription::BaseOpsBool(ops) => ops.cleanup_tensor(handles),
            TensorOpsDescription::NumericOpsFloat(ops) => ops.cleanup_tensor(handles),
            TensorOpsDescription::NumericOpsInt(ops) => ops.cleanup_tensor(handles),
            TensorOpsDescription::BoolOps(ops) => ops.cleanup_tensor(handles),
            TensorOpsDescription::IntOps(ops) => ops.cleanup_tensor(handles),
            TensorOpsDescription::FloatOps(ops) => ops.cleanup_tensor(handles),
            TensorOpsDescription::ModuleOps(_) => todo!(),
        }
    }
    pub(crate) fn execute(&self, handles: &mut HandleContainer<B>) {
        match self {
            TensorOpsDescription::BaseOpsFloat(ops) => ops.execute(handles),
            TensorOpsDescription::BaseOpsInt(ops) => ops.execute(handles),
            TensorOpsDescription::BaseOpsBool(ops) => ops.execute(handles),
            TensorOpsDescription::NumericOpsFloat(ops) => ops.execute(handles),
            TensorOpsDescription::NumericOpsInt(ops) => ops.execute(handles),
            TensorOpsDescription::BoolOps(ops) => ops.execute(handles),
            TensorOpsDescription::IntOps(ops) => ops.execute(handles),
            TensorOpsDescription::FloatOps(ops) => ops.execute(handles),
            TensorOpsDescription::ModuleOps(_) => todo!(),
        }
    }
}

impl<B: FusedBackend> BaseOpsDescription<B> {
    fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        match self {
            BaseOpsDescription::ToDevice(_, _) => (),
            BaseOpsDescription::Reshape(desc, _) => {
                handles.cleanup(&desc.input);
            }
            BaseOpsDescription::SwapDims(desc, _) => {
                handles.cleanup(&desc.input);
            }
            BaseOpsDescription::Slice(desc, _) => {
                handles.cleanup(&desc.tensor);
            }
            BaseOpsDescription::SliceAssign(desc, _) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.value);
            }
            BaseOpsDescription::Equal(desc, _) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            BaseOpsDescription::Repeat(desc, _) => {
                handles.cleanup(&desc.tensor);
            }
            BaseOpsDescription::Cat(desc, _) => {
                for t in desc.tensors.iter() {
                    handles.cleanup(t);
                }
            }
        }
    }
    fn execute(&self, handles: &mut HandleContainer<B>) {
        match self {
            BaseOpsDescription::ToDevice(desc, ops) => ops.execute(desc, handles),
            BaseOpsDescription::Reshape(desc, ops) => ops.execute(desc, handles),
            BaseOpsDescription::SwapDims(desc, ops) => ops.execute(desc, handles),
            BaseOpsDescription::Slice(desc, ops) => ops.execute(desc, handles),
            BaseOpsDescription::SliceAssign(desc, ops) => ops.execute(desc, handles),
            BaseOpsDescription::Equal(desc, ops) => ops.execute(desc, handles),
            BaseOpsDescription::Repeat(desc, ops) => ops.execute(desc, handles),
            BaseOpsDescription::Cat(desc, ops) => ops.execute(desc, handles),
        }
    }
}

impl<B: FusedBackend, E: Element> NumericOpsDescription<B, E> {
    fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        match self {
            NumericOpsDescription::Add(desc, ops) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::AddScalar(desc, ops) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Sub(desc, ops) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::SubScalar(desc, ops) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Mul(desc, ops) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::MulScalar(desc, ops) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Div(desc, ops) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::DivScalar(desc, ops) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Ones(desc, ops) => {}
            NumericOpsDescription::Gather(desc, ops) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.indices);
            }
            NumericOpsDescription::Scatter(desc, ops) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.indices);
                handles.cleanup(&desc.value);
            }
            NumericOpsDescription::Select(desc, ops) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.indices);
            }
            NumericOpsDescription::SelectAssign(desc, ops) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.indices);
                handles.cleanup(&desc.value);
            }
            NumericOpsDescription::MaskWhere(desc, ops) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.value);
                handles.cleanup(&desc.mask);
            }
            NumericOpsDescription::MaskFill(desc, ops) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.mask);
            }
            NumericOpsDescription::EqualElem(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::GreaterElem(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::GreaterEqualElem(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::LowerElem(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::LowerEqualElem(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Greater(desc, _) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::GreaterEqual(desc, _) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::Lower(desc, _) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::LowerEqual(desc, _) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::ArgMax(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::ArgMin(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Clamp(desc, _) => {
                handles.cleanup(&desc.tensor);
            }
            NumericOpsDescription::ClampMin(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::ClampMax(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Abs(desc, _) => {
                handles.cleanup(&desc.input);
            }
            NumericOpsDescription::Zeros(desc, _) => {}
            NumericOpsDescription::Full(desc, _) => {}
            NumericOpsDescription::MeanDim(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Mean(desc, _) => {
                handles.cleanup(&desc.input);
            }
            NumericOpsDescription::Sum(desc, _) => {
                handles.cleanup(&desc.input);
            }
            NumericOpsDescription::SumDim(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Max(desc, _) => {
                handles.cleanup(&desc.input);
            }
            NumericOpsDescription::MaxDimWithIndices(desc, _) => {
                handles.cleanup(&desc.tensor);
            }
            NumericOpsDescription::MinDimWithIndices(desc, _) => {
                handles.cleanup(&desc.tensor);
            }
            NumericOpsDescription::Min(desc, _) => {
                handles.cleanup(&desc.input);
            }
            NumericOpsDescription::MaxDim(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::MinDim(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
        }
    }

    fn execute(&self, handles: &mut HandleContainer<B>) {
        match self {
            NumericOpsDescription::Add(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::AddScalar(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Sub(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::SubScalar(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Div(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::DivScalar(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Mul(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::MulScalar(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Ones(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Gather(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Scatter(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Select(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::SelectAssign(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::MaskWhere(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::MaskFill(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::EqualElem(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Greater(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::GreaterElem(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::GreaterEqual(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::GreaterEqualElem(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Lower(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::LowerElem(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::LowerEqual(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::LowerEqualElem(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::ArgMax(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::ArgMin(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Clamp(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::ClampMin(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::ClampMax(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Abs(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Zeros(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Full(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::MeanDim(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Mean(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Sum(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::SumDim(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Max(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::MaxDimWithIndices(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::MinDimWithIndices(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Min(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::MaxDim(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::MinDim(desc, ops) => ops.execute(desc, handles),
        }
    }
}

impl<B: FusedBackend> FloatOpsDescription<B> {
    fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        match self {
            FloatOpsDescription::Matmul(desc, ops) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            FloatOpsDescription::Random(desc, ops) => {}
            FloatOpsDescription::Exp(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::Log(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::Log1p(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::Erf(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::Powf(desc, _) => handles.cleanup(&desc.lhs),
            FloatOpsDescription::Sqrt(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::Cos(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::Sin(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::Tanh(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::IntoInt(desc, _) => handles.cleanup(&desc.input),
        }
    }
    fn execute(&self, handles: &mut HandleContainer<B>) {
        match self {
            FloatOpsDescription::Matmul(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Random(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Exp(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Log(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Log1p(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Erf(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Powf(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Sqrt(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Cos(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Sin(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Tanh(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::IntoInt(desc, ops) => ops.execute(desc, handles),
        }
    }
}

impl<B: FusedBackend> IntOpsDescription<B> {
    fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        match self {
            IntOpsDescription::IntoFloat(desc, _) => {
                handles.cleanup(&desc.input);
            }
        }
    }
    fn execute(&self, handles: &mut HandleContainer<B>) {
        match self {
            IntOpsDescription::IntoFloat(desc, ops) => ops.execute(desc, handles),
        }
    }
}

impl<B: FusedBackend> BoolOpsDescription<B> {
    fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        match self {
            BoolOpsDescription::IntoFloat(desc, _) => {
                handles.cleanup(&desc.input);
            }
            BoolOpsDescription::IntoInt(desc, _) => {
                handles.cleanup(&desc.input);
            }
            BoolOpsDescription::Not(desc, _) => {
                handles.cleanup(&desc.input);
            }
        }
    }
    fn execute(&self, handles: &mut HandleContainer<B>) {
        match self {
            BoolOpsDescription::IntoFloat(desc, ops) => ops.execute(desc, handles),
            BoolOpsDescription::IntoInt(desc, ops) => ops.execute(desc, handles),
            BoolOpsDescription::Not(desc, ops) => ops.execute(desc, handles),
        }
    }
}

pub enum FloatOpsDescription<B: FusedBackend> {
    Exp(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    Log(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    Log1p(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    Erf(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    Powf(
        ScalarOpsDescription<f32>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<f32>>>,
    ),
    Sqrt(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    Cos(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    Sin(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    Tanh(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    IntoInt(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    Matmul(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    Random(
        (TensorDescription, Distribution<FloatElem<B>>),
        Box<dyn Ops<B, Args = (TensorDescription, Distribution<FloatElem<B>>)>>,
    ),
}

#[derive(Debug)]
pub enum ModuleOpsDescription {
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

pub enum BaseOpsDescription<B: FusedBackend> {
    ToDevice(
        (TensorDescription, B::Device),
        Box<dyn Ops<B, Args = (TensorDescription, B::Device)>>,
    ),
    Reshape(
        ReshapeDescription,
        Box<dyn Ops<B, Args = ReshapeDescription>>,
    ),
    SwapDims(
        SwapDimsDescription,
        Box<dyn Ops<B, Args = SwapDimsDescription>>,
    ),
    Slice(
        SliceOpsDescription,
        Box<dyn Ops<B, Args = SliceOpsDescription>>,
    ),
    SliceAssign(
        SliceAssignOpsDescription,
        Box<dyn Ops<B, Args = SliceAssignOpsDescription>>,
    ),
    Equal(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    Repeat(
        RepeatOpsDescription,
        Box<dyn Ops<B, Args = RepeatOpsDescription>>,
    ),
    Cat(CatOpsDescription, Box<dyn Ops<B, Args = CatOpsDescription>>),
}

pub trait Ops<B: FusedBackend>: Send + Sync {
    type Args: Send + Sync;

    fn execute(&self, args: &Self::Args, handles: &mut HandleContainer<B>);
}

pub enum NumericOpsDescription<B: FusedBackend, E: Element> {
    Add(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    AddScalar(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    Sub(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    SubScalar(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    Div(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    DivScalar(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    Mul(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    MulScalar(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    Abs(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    Ones(TensorDescription, Box<dyn Ops<B, Args = TensorDescription>>),
    Zeros(TensorDescription, Box<dyn Ops<B, Args = TensorDescription>>),
    Full(
        (TensorDescription, E),
        Box<dyn Ops<B, Args = (TensorDescription, E)>>,
    ),
    Gather(
        GatherOpsDescription,
        Box<dyn Ops<B, Args = GatherOpsDescription>>,
    ),
    Scatter(
        ScatterOpsDescription,
        Box<dyn Ops<B, Args = ScatterOpsDescription>>,
    ),
    Select(
        SelectOpsDescription,
        Box<dyn Ops<B, Args = SelectOpsDescription>>,
    ),
    SelectAssign(
        SelectAssignOpsDescription,
        Box<dyn Ops<B, Args = SelectAssignOpsDescription>>,
    ),
    MaskWhere(
        MaskWhereOpsDescription,
        Box<dyn Ops<B, Args = MaskWhereOpsDescription>>,
    ),
    MaskFill(
        MaskFillOpsDescription<E>,
        Box<dyn Ops<B, Args = MaskFillOpsDescription<E>>>,
    ),
    MeanDim(
        ScalarOpsDescription<usize>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<usize>>>,
    ),
    Mean(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    Sum(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    SumDim(
        ScalarOpsDescription<usize>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<usize>>>,
    ),
    EqualElem(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    Greater(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    GreaterElem(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    GreaterEqual(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    GreaterEqualElem(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    Lower(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    LowerElem(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    LowerEqual(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    LowerEqualElem(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    ArgMax(
        ScalarOpsDescription<usize>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<usize>>>,
    ),
    ArgMin(
        ScalarOpsDescription<usize>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<usize>>>,
    ),
    Max(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    MaxDimWithIndices(
        ReduceDimWithIndicesDescription,
        Box<dyn Ops<B, Args = ReduceDimWithIndicesDescription>>,
    ),
    MinDimWithIndices(
        ReduceDimWithIndicesDescription,
        Box<dyn Ops<B, Args = ReduceDimWithIndicesDescription>>,
    ),
    Min(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    MaxDim(
        ScalarOpsDescription<usize>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<usize>>>,
    ),
    MinDim(
        ScalarOpsDescription<usize>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<usize>>>,
    ),

    Clamp(
        ClampOpsDescription<E>,
        Box<dyn Ops<B, Args = ClampOpsDescription<E>>>,
    ),
    ClampMax(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    ClampMin(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
}

pub enum IntOpsDescription<B: FusedBackend> {
    IntoFloat(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
}

pub enum BoolOpsDescription<B: FusedBackend> {
    IntoFloat(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    IntoInt(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    Not(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
}

pub struct SwapDimsDescription {
    pub input: TensorDescription,
    pub out: TensorDescription,
    pub dim1: usize,
    pub dim2: usize,
}

pub struct ReshapeDescription {
    pub input: TensorDescription,
    pub out: TensorDescription,
    pub shape: Vec<usize>,
}

pub struct BinaryOpsDescription {
    pub lhs: TensorDescription,
    pub rhs: TensorDescription,
    pub out: TensorDescription,
}

pub struct UnaryOpsDescription {
    pub input: TensorDescription,
    pub out: TensorDescription,
}

pub struct ScalarOpsDescription<E> {
    pub lhs: TensorDescription,
    pub rhs: E,
    pub out: TensorDescription,
}

pub struct GatherOpsDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub indices: TensorDescription,
    pub out: TensorDescription,
}

pub struct ScatterOpsDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub indices: TensorDescription,
    pub value: TensorDescription,
    pub out: TensorDescription,
}

pub struct SelectOpsDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub indices: TensorDescription,
    pub out: TensorDescription,
}

pub struct SelectAssignOpsDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub indices: TensorDescription,
    pub value: TensorDescription,
    pub out: TensorDescription,
}

pub struct SliceOpsDescription {
    pub tensor: TensorDescription,
    pub ranges: Vec<Range<usize>>,
    pub out: TensorDescription,
}

pub struct SliceAssignOpsDescription {
    pub tensor: TensorDescription,
    pub ranges: Vec<Range<usize>>,
    pub value: TensorDescription,
    pub out: TensorDescription,
}
pub struct MaskWhereOpsDescription {
    pub tensor: TensorDescription,
    pub mask: TensorDescription,
    pub value: TensorDescription,
    pub out: TensorDescription,
}

pub struct MaskFillOpsDescription<E> {
    pub tensor: TensorDescription,
    pub mask: TensorDescription,
    pub value: E,
    pub out: TensorDescription,
}

pub struct ClampOpsDescription<E> {
    pub tensor: TensorDescription,
    pub min: E,
    pub max: E,
    pub out: TensorDescription,
}
pub struct RepeatOpsDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub times: usize,
    pub shape: Vec<usize>,
    pub out: TensorDescription,
}

pub struct CatOpsDescription {
    pub tensors: Vec<TensorDescription>,
    pub dim: usize,
    pub out: TensorDescription,
}

pub struct ReduceDimWithIndicesDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub out: TensorDescription,
    pub out_indices: TensorDescription,
}
