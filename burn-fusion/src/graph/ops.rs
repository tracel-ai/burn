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
    ModuleOps(ModuleOpsDescription<B>),
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
            TensorOpsDescription::ModuleOps(ops) => ops.cleanup_tensor(handles),
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
            TensorOpsDescription::ModuleOps(ops) => ops.execute(handles),
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

impl<B: FusedBackend> ModuleOpsDescription<B> {
    fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        match self {
            ModuleOpsDescription::Embedding(desc, _) => {
                handles.cleanup(&desc.weights);
                handles.cleanup(&desc.indices);
            }
            ModuleOpsDescription::EmbeddingBackward(desc, _) => {
                handles.cleanup(&desc.weights);
                handles.cleanup(&desc.out_grad);
                handles.cleanup(&desc.indices);
            }
            ModuleOpsDescription::Conv1d(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.weight);

                if let Some(bias) = &desc.bias {
                    handles.cleanup(bias);
                }
            }
            ModuleOpsDescription::Conv2d(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.weight);

                if let Some(bias) = &desc.bias {
                    handles.cleanup(bias);
                }
            }
            ModuleOpsDescription::ConvTranspose1d(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.weight);

                if let Some(bias) = &desc.bias {
                    handles.cleanup(bias);
                }
            }
            ModuleOpsDescription::ConvTranspose2d(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.weight);

                if let Some(bias) = &desc.bias {
                    handles.cleanup(bias);
                }
            }
            ModuleOpsDescription::AvgPool1d(desc, _) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::AvgPool2d(desc, _) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::AvgPool1dBackward(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
            }
            ModuleOpsDescription::AvgPool2dBackward(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
            }
            ModuleOpsDescription::AdaptiveAvgPool1d(desc, _) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::AdaptiveAvgPool2d(desc, _) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::AdaptiveAvgPool1dBackward(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
            }
            ModuleOpsDescription::AdaptiveAvgPool2dBackward(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
            }
            ModuleOpsDescription::MaxPool1d(desc, _) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::MaxPool1dWithIndices(desc, _) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::MaxPool1dWithIndicesBackward(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
                handles.cleanup(&desc.indices);
            }
            ModuleOpsDescription::MaxPool2d(desc, _) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::MaxPool2dWithIndices(desc, _) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::MaxPool2dWithIndicesBackward(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
                handles.cleanup(&desc.indices);
            }
        }
    }
    fn execute(&self, handles: &mut HandleContainer<B>) {
        match self {
            ModuleOpsDescription::Embedding(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::EmbeddingBackward(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::Conv1d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::Conv2d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::ConvTranspose1d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::ConvTranspose2d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::AvgPool1d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::AvgPool2d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::AvgPool1dBackward(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::AvgPool2dBackward(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::AdaptiveAvgPool1d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::AdaptiveAvgPool2d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::AdaptiveAvgPool1dBackward(desc, ops) => {
                ops.execute(desc, handles)
            }
            ModuleOpsDescription::AdaptiveAvgPool2dBackward(desc, ops) => {
                ops.execute(desc, handles)
            }
            ModuleOpsDescription::MaxPool1d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::MaxPool1dWithIndices(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::MaxPool1dWithIndicesBackward(desc, ops) => {
                ops.execute(desc, handles)
            }
            ModuleOpsDescription::MaxPool2d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::MaxPool2dWithIndices(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::MaxPool2dWithIndicesBackward(desc, ops) => {
                ops.execute(desc, handles)
            }
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

pub enum ModuleOpsDescription<B: FusedBackend> {
    Embedding(
        EmbeddingDescription,
        Box<dyn Ops<B, Args = EmbeddingDescription>>,
    ),
    EmbeddingBackward(
        EmbeddingBackwardDescription,
        Box<dyn Ops<B, Args = EmbeddingBackwardDescription>>,
    ),
    Conv1d(Conv1dDescription, Box<dyn Ops<B, Args = Conv1dDescription>>),
    Conv2d(Conv2dDescription, Box<dyn Ops<B, Args = Conv2dDescription>>),
    ConvTranspose1d(
        ConvTranspose1dDescription,
        Box<dyn Ops<B, Args = ConvTranspose1dDescription>>,
    ),
    ConvTranspose2d(
        ConvTranspose2dDescription,
        Box<dyn Ops<B, Args = ConvTranspose2dDescription>>,
    ),
    AvgPool1d(
        AvgPool1dDescription,
        Box<dyn Ops<B, Args = AvgPool1dDescription>>,
    ),
    AvgPool2d(
        AvgPool2dDescription,
        Box<dyn Ops<B, Args = AvgPool2dDescription>>,
    ),
    AvgPool1dBackward(
        AvgPool1dBackwardDescription,
        Box<dyn Ops<B, Args = AvgPool1dBackwardDescription>>,
    ),
    AvgPool2dBackward(
        AvgPool2dBackwardDescription,
        Box<dyn Ops<B, Args = AvgPool2dBackwardDescription>>,
    ),
    AdaptiveAvgPool1d(
        AdaptiveAvgPool1dDescription,
        Box<dyn Ops<B, Args = AdaptiveAvgPool1dDescription>>,
    ),
    AdaptiveAvgPool2d(
        AdaptiveAvgPool2dDescription,
        Box<dyn Ops<B, Args = AdaptiveAvgPool2dDescription>>,
    ),
    AdaptiveAvgPool1dBackward(
        AdaptiveAvgPool1dBackwardDescription,
        Box<dyn Ops<B, Args = AdaptiveAvgPool1dBackwardDescription>>,
    ),
    AdaptiveAvgPool2dBackward(
        AdaptiveAvgPool2dBackwardDescription,
        Box<dyn Ops<B, Args = AdaptiveAvgPool2dBackwardDescription>>,
    ),
    MaxPool1d(
        MaxPool1dDescription,
        Box<dyn Ops<B, Args = MaxPool1dDescription>>,
    ),
    MaxPool1dWithIndices(
        MaxPool1dWithIndicesDescription,
        Box<dyn Ops<B, Args = MaxPool1dWithIndicesDescription>>,
    ),
    MaxPool1dWithIndicesBackward(
        MaxPool1dWithIndicesBackwardDescription,
        Box<dyn Ops<B, Args = MaxPool1dWithIndicesBackwardDescription>>,
    ),
    MaxPool2d(
        MaxPool2dDescription,
        Box<dyn Ops<B, Args = MaxPool2dDescription>>,
    ),
    MaxPool2dWithIndices(
        MaxPool2dWithIndicesDescription,
        Box<dyn Ops<B, Args = MaxPool2dWithIndicesDescription>>,
    ),
    MaxPool2dWithIndicesBackward(
        MaxPool2dWithIndicesBackwardDescription,
        Box<dyn Ops<B, Args = MaxPool2dWithIndicesBackwardDescription>>,
    ),
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

pub struct EmbeddingDescription {
    pub weights: TensorDescription,
    pub indices: TensorDescription,
    pub out: TensorDescription,
}

pub struct EmbeddingBackwardDescription {
    pub weights: TensorDescription,
    pub out_grad: TensorDescription,
    pub indices: TensorDescription,
    pub out: TensorDescription,
}

pub struct Conv1dDescription {
    pub x: TensorDescription,
    pub weight: TensorDescription,
    pub bias: Option<TensorDescription>,
    pub options: ConvOptions<1>,
    pub out: TensorDescription,
}

pub struct Conv2dDescription {
    pub x: TensorDescription,
    pub weight: TensorDescription,
    pub bias: Option<TensorDescription>,
    pub options: ConvOptions<2>,
    pub out: TensorDescription,
}

pub struct ConvTranspose1dDescription {
    pub x: TensorDescription,
    pub weight: TensorDescription,
    pub bias: Option<TensorDescription>,
    pub options: ConvTransposeOptions<1>,
    pub out: TensorDescription,
}

pub struct ConvTranspose2dDescription {
    pub x: TensorDescription,
    pub weight: TensorDescription,
    pub bias: Option<TensorDescription>,
    pub options: ConvTransposeOptions<2>,
    pub out: TensorDescription,
}

pub struct AvgPool1dDescription {
    pub x: TensorDescription,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub count_include_pad: bool,
    pub out: TensorDescription,
}

pub struct AvgPool2dDescription {
    pub x: TensorDescription,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub count_include_pad: bool,
    pub out: TensorDescription,
}

pub struct AvgPool1dBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub count_include_pad: bool,
    pub out: TensorDescription,
}

pub struct AvgPool2dBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub count_include_pad: bool,
    pub out: TensorDescription,
}

pub struct AdaptiveAvgPool1dDescription {
    pub x: TensorDescription,
    pub output_size: usize,
    pub out: TensorDescription,
}

pub struct AdaptiveAvgPool2dDescription {
    pub x: TensorDescription,
    pub output_size: [usize; 2],
    pub out: TensorDescription,
}

pub struct AdaptiveAvgPool1dBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub out: TensorDescription,
}

pub struct AdaptiveAvgPool2dBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub out: TensorDescription,
}

pub struct MaxPool1dDescription {
    pub x: TensorDescription,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out: TensorDescription,
}

pub struct MaxPool1dWithIndicesDescription {
    pub x: TensorDescription,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out: TensorDescription,
    pub out_indices: TensorDescription,
}

pub struct MaxPool1dWithIndicesBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub indices: TensorDescription,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out: TensorDescription,
}

pub struct MaxPool2dDescription {
    pub x: TensorDescription,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub out: TensorDescription,
}

pub struct MaxPool2dWithIndicesDescription {
    pub x: TensorDescription,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub out: TensorDescription,
    pub out_indices: TensorDescription,
}

pub struct MaxPool2dWithIndicesBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub indices: TensorDescription,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub out: TensorDescription,
}
