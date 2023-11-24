use super::{
    AdaptiveAvgPool1dBackwardDescription, AdaptiveAvgPool1dDescription,
    AdaptiveAvgPool2dBackwardDescription, AdaptiveAvgPool2dDescription,
    AvgPool2dBackwardDescription, AvgPool2dDescription, BaseOpsDescription, BinaryOpsDescription,
    BoolOpsDescription, ClampOpsDescription, Conv1dDescription, Conv2dDescription,
    ConvTranspose1dDescription, ConvTranspose2dDescription, EmbeddingBackwardDescription,
    EmbeddingDescription, FloatOpsDescription, GatherOpsDescription, IntOpsDescription,
    MaskFillOpsDescription, MaskWhereOpsDescription, MaxPool1dDescription,
    MaxPool1dWithIndicesBackwardDescription, MaxPool1dWithIndicesDescription, MaxPool2dDescription,
    MaxPool2dWithIndicesBackwardDescription, MaxPool2dWithIndicesDescription, ModuleOpsDescription,
    NumericOpsDescription, RandomOpsDescription, ReduceDimWithIndicesDescription,
    ReshapeDescription, ScalarOpsDescription, ScatterOpsDescription, SelectAssignOpsDescription,
    SelectOpsDescription, SliceOpsDescription, SwapDimsDescription, TensorOpsDescription,
    UnaryOpsDescription,
};
use crate::{FusionBackend, HandleContainer, TensorDescription, TensorId};
use burn_tensor::{Element, ElementConversion};
use hashbrown::HashMap;

#[derive(new)]
pub struct Context<'a, 'b, B: FusionBackend> {
    pub tensors: &'a HashMap<TensorId, TensorDescription>,
    pub handles: &'b mut HandleContainer<B>,
    pub scalar_floats: &'a Vec<f32>,
    pub scalar_ints: &'a Vec<i32>,
}

#[derive(Default)]
pub(crate) struct LocalGraphConverter {
    tensors_local2global: HashMap<TensorId, TensorDescription>,
    tensors_global2local: HashMap<TensorId, TensorDescription>,
    /// Only useful to create new shape ID.
    /// You should use tensor descriptions to retrieve the proper shape.
    shapes_local2global: HashMap<usize, usize>,
    scalar_floats: Vec<f32>,
    scalar_ints: Vec<i32>,
}

impl LocalGraphConverter {
    pub(crate) fn context<'a, 'b, B: FusionBackend>(
        &'a self,
        handles: &'b mut HandleContainer<B>,
    ) -> Context<'a, 'b, B> {
        Context {
            handles,
            tensors: &self.tensors_local2global,
            scalar_floats: &self.scalar_floats,
            scalar_ints: &self.scalar_ints,
        }
    }
    pub(crate) fn clear(&mut self) {
        self.tensors_local2global.clear();
        self.tensors_global2local.clear();
        self.shapes_local2global.clear();
    }
    pub(crate) fn local_float<E: Element>(&mut self, elem: &E) -> E {
        self.scalar_floats.push(elem.elem());
        0.elem()
    }
    pub(crate) fn local_int<E: Element>(&mut self, elem: &E) -> E {
        self.scalar_floats.push(elem.elem());
        0.elem()
    }
}

impl TensorOpsDescription {
    pub(crate) fn to_local(&self, converter: &mut LocalGraphConverter) -> Self {
        match self {
            TensorOpsDescription::BaseOpsFloat(ops) => {
                TensorOpsDescription::BaseOpsFloat(ops.to_local(converter))
            }
            TensorOpsDescription::BaseOpsInt(ops) => {
                TensorOpsDescription::BaseOpsInt(ops.to_local(converter))
            }
            TensorOpsDescription::BaseOpsBool(ops) => {
                TensorOpsDescription::BaseOpsBool(ops.to_local(converter))
            }
            TensorOpsDescription::NumericOpsFloat(ops) => TensorOpsDescription::NumericOpsFloat(
                ops.to_local(converter, |converter, e| converter.local_float(e)),
            ),
            TensorOpsDescription::NumericOpsInt(ops) => TensorOpsDescription::NumericOpsInt(
                ops.to_local(converter, |converter, e| converter.local_int(e)),
            ),
            TensorOpsDescription::BoolOps(ops) => {
                TensorOpsDescription::BoolOps(ops.to_local(converter))
            }
            TensorOpsDescription::IntOps(ops) => {
                TensorOpsDescription::IntOps(ops.to_local(converter))
            }
            TensorOpsDescription::FloatOps(ops) => {
                TensorOpsDescription::FloatOps(ops.to_local(converter))
            }
            TensorOpsDescription::ModuleOps(ops) => {
                TensorOpsDescription::ModuleOps(ops.to_local(converter))
            }
        }
    }
}
impl ModuleOpsDescription {
    pub(crate) fn to_local(&self, converter: &mut LocalGraphConverter) -> Self {
        match self {
            ModuleOpsDescription::Embedding(desc) => {
                ModuleOpsDescription::Embedding(EmbeddingDescription {
                    weights: desc.weights.to_local(converter),
                    indices: desc.indices.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            ModuleOpsDescription::EmbeddingBackward(desc) => {
                ModuleOpsDescription::EmbeddingBackward(EmbeddingBackwardDescription {
                    weights: desc.weights.to_local(converter),
                    out_grad: desc.out_grad.to_local(converter),
                    indices: desc.indices.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            ModuleOpsDescription::Conv1d(desc) => ModuleOpsDescription::Conv1d(Conv1dDescription {
                x: desc.x.to_local(converter),
                weight: desc.weight.to_local(converter),
                bias: desc.bias.as_ref().map(|t| t.to_local(converter)),
                options: desc.options.clone(),
                out: desc.out.to_local(converter),
            }),
            ModuleOpsDescription::Conv2d(desc) => ModuleOpsDescription::Conv2d(Conv2dDescription {
                x: desc.x.to_local(converter),
                weight: desc.weight.to_local(converter),
                bias: desc.bias.as_ref().map(|t| t.to_local(converter)),
                options: desc.options.clone(),
                out: desc.out.to_local(converter),
            }),
            ModuleOpsDescription::ConvTranspose1d(desc) => {
                ModuleOpsDescription::ConvTranspose1d(ConvTranspose1dDescription {
                    x: desc.x.to_local(converter),
                    weight: desc.weight.to_local(converter),
                    bias: desc.bias.as_ref().map(|t| t.to_local(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_local(converter),
                })
            }
            ModuleOpsDescription::ConvTranspose2d(desc) => {
                ModuleOpsDescription::ConvTranspose2d(ConvTranspose2dDescription {
                    x: desc.x.to_local(converter),
                    weight: desc.weight.to_local(converter),
                    bias: desc.bias.as_ref().map(|t| t.to_local(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_local(converter),
                })
            }
            ModuleOpsDescription::AvgPool1d(desc) => {
                ModuleOpsDescription::AvgPool1d(super::AvgPool1dDescription {
                    x: desc.x.to_local(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_local(converter),
                })
            }
            ModuleOpsDescription::AvgPool2d(desc) => {
                ModuleOpsDescription::AvgPool2d(AvgPool2dDescription {
                    x: desc.x.to_local(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_local(converter),
                })
            }
            ModuleOpsDescription::AvgPool1dBackward(desc) => {
                ModuleOpsDescription::AvgPool1dBackward(super::AvgPool1dBackwardDescription {
                    x: desc.x.to_local(converter),
                    grad: desc.grad.to_local(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_local(converter),
                })
            }
            ModuleOpsDescription::AvgPool2dBackward(desc) => {
                ModuleOpsDescription::AvgPool2dBackward(AvgPool2dBackwardDescription {
                    x: desc.x.to_local(converter),
                    grad: desc.grad.to_local(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_local(converter),
                })
            }
            ModuleOpsDescription::AdaptiveAvgPool1d(desc) => {
                ModuleOpsDescription::AdaptiveAvgPool1d(AdaptiveAvgPool1dDescription {
                    x: desc.x.to_local(converter),
                    output_size: desc.output_size,
                    out: desc.out.to_local(converter),
                })
            }
            ModuleOpsDescription::AdaptiveAvgPool2d(desc) => {
                ModuleOpsDescription::AdaptiveAvgPool2d(AdaptiveAvgPool2dDescription {
                    x: desc.x.to_local(converter),
                    output_size: desc.output_size,
                    out: desc.out.to_local(converter),
                })
            }
            ModuleOpsDescription::AdaptiveAvgPool1dBackward(desc) => {
                ModuleOpsDescription::AdaptiveAvgPool1dBackward(
                    AdaptiveAvgPool1dBackwardDescription {
                        x: desc.x.to_local(converter),
                        grad: desc.grad.to_local(converter),
                        out: desc.out.to_local(converter),
                    },
                )
            }
            ModuleOpsDescription::AdaptiveAvgPool2dBackward(desc) => {
                ModuleOpsDescription::AdaptiveAvgPool2dBackward(
                    AdaptiveAvgPool2dBackwardDescription {
                        x: desc.x.to_local(converter),
                        grad: desc.grad.to_local(converter),
                        out: desc.out.to_local(converter),
                    },
                )
            }
            ModuleOpsDescription::MaxPool1d(desc) => {
                ModuleOpsDescription::MaxPool1d(MaxPool1dDescription {
                    x: desc.x.to_local(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_local(converter),
                })
            }
            ModuleOpsDescription::MaxPool1dWithIndices(desc) => {
                ModuleOpsDescription::MaxPool1dWithIndices(MaxPool1dWithIndicesDescription {
                    x: desc.x.to_local(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_local(converter),
                    out_indices: desc.out_indices.to_local(converter),
                })
            }
            ModuleOpsDescription::MaxPool1dWithIndicesBackward(desc) => {
                ModuleOpsDescription::MaxPool1dWithIndicesBackward(
                    MaxPool1dWithIndicesBackwardDescription {
                        x: desc.x.to_local(converter),
                        grad: desc.grad.to_local(converter),
                        indices: desc.indices.to_local(converter),
                        kernel_size: desc.kernel_size,
                        stride: desc.stride,
                        padding: desc.padding,
                        dilation: desc.dilation,
                        out: desc.out.to_local(converter),
                    },
                )
            }
            ModuleOpsDescription::MaxPool2d(desc) => {
                ModuleOpsDescription::MaxPool2d(MaxPool2dDescription {
                    x: desc.x.to_local(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_local(converter),
                })
            }
            ModuleOpsDescription::MaxPool2dWithIndices(desc) => {
                ModuleOpsDescription::MaxPool2dWithIndices(MaxPool2dWithIndicesDescription {
                    x: desc.x.to_local(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_local(converter),
                    out_indices: desc.out_indices.to_local(converter),
                })
            }
            ModuleOpsDescription::MaxPool2dWithIndicesBackward(desc) => {
                ModuleOpsDescription::MaxPool2dWithIndicesBackward(
                    MaxPool2dWithIndicesBackwardDescription {
                        x: desc.x.to_local(converter),
                        grad: desc.grad.to_local(converter),
                        indices: desc.indices.to_local(converter),
                        kernel_size: desc.kernel_size,
                        stride: desc.stride,
                        padding: desc.padding,
                        dilation: desc.dilation,
                        out: desc.out.to_local(converter),
                    },
                )
            }
        }
    }
}

impl FloatOpsDescription {
    pub(crate) fn to_local(&self, converter: &mut LocalGraphConverter) -> Self {
        match self {
            FloatOpsDescription::Exp(desc) => FloatOpsDescription::Exp(UnaryOpsDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            FloatOpsDescription::Log(desc) => FloatOpsDescription::Log(UnaryOpsDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            FloatOpsDescription::Log1p(desc) => FloatOpsDescription::Log1p(UnaryOpsDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            FloatOpsDescription::Erf(desc) => FloatOpsDescription::Erf(UnaryOpsDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            FloatOpsDescription::Powf(desc) => FloatOpsDescription::Powf(ScalarOpsDescription {
                lhs: desc.lhs.to_local(converter),
                rhs: converter.local_float(&desc.rhs),
                out: desc.out.to_local(converter),
            }),
            FloatOpsDescription::Sqrt(desc) => FloatOpsDescription::Sqrt(UnaryOpsDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            FloatOpsDescription::Cos(desc) => FloatOpsDescription::Cos(UnaryOpsDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            FloatOpsDescription::Sin(desc) => FloatOpsDescription::Sin(UnaryOpsDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            FloatOpsDescription::Tanh(desc) => FloatOpsDescription::Tanh(UnaryOpsDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            FloatOpsDescription::IntoInt(desc) => {
                FloatOpsDescription::IntoInt(UnaryOpsDescription {
                    input: desc.input.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            FloatOpsDescription::Matmul(desc) => {
                FloatOpsDescription::Matmul(BinaryOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: desc.rhs.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            FloatOpsDescription::Random(desc) => {
                FloatOpsDescription::Random(RandomOpsDescription {
                    out: desc.out.to_local(converter),
                    distribution: desc.distribution,
                })
            }
            FloatOpsDescription::Recip(desc) => FloatOpsDescription::Recip(UnaryOpsDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
        }
    }
}

impl BoolOpsDescription {
    pub(crate) fn to_local(&self, converter: &mut LocalGraphConverter) -> Self {
        match self {
            BoolOpsDescription::IntoFloat(desc) => {
                BoolOpsDescription::IntoFloat(UnaryOpsDescription {
                    input: desc.input.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            BoolOpsDescription::IntoInt(desc) => BoolOpsDescription::IntoInt(UnaryOpsDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            BoolOpsDescription::Not(desc) => BoolOpsDescription::Not(UnaryOpsDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
        }
    }
}

impl IntOpsDescription {
    pub(crate) fn to_local(&self, converter: &mut LocalGraphConverter) -> Self {
        match self {
            IntOpsDescription::IntoFloat(desc) => {
                IntOpsDescription::IntoFloat(UnaryOpsDescription {
                    input: desc.input.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
        }
    }
}

impl<E: Element> NumericOpsDescription<E> {
    pub(crate) fn to_local<F>(&self, converter: &mut LocalGraphConverter, local_elem: F) -> Self
    where
        F: Fn(&mut LocalGraphConverter, &E) -> E,
    {
        match self {
            NumericOpsDescription::Add(desc) => NumericOpsDescription::Add(BinaryOpsDescription {
                lhs: desc.lhs.to_local(converter),
                rhs: desc.rhs.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            NumericOpsDescription::AddScalar(desc) => {
                NumericOpsDescription::AddScalar(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::Sub(desc) => NumericOpsDescription::Sub(BinaryOpsDescription {
                lhs: desc.lhs.to_local(converter),
                rhs: desc.rhs.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            NumericOpsDescription::SubScalar(desc) => {
                NumericOpsDescription::SubScalar(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::Div(desc) => NumericOpsDescription::Div(BinaryOpsDescription {
                lhs: desc.lhs.to_local(converter),
                rhs: desc.rhs.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            NumericOpsDescription::DivScalar(desc) => {
                NumericOpsDescription::DivScalar(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::Mul(desc) => NumericOpsDescription::Mul(BinaryOpsDescription {
                lhs: desc.lhs.to_local(converter),
                rhs: desc.rhs.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            NumericOpsDescription::MulScalar(desc) => {
                NumericOpsDescription::MulScalar(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::Abs(desc) => NumericOpsDescription::Abs(UnaryOpsDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            NumericOpsDescription::Ones(desc) => {
                NumericOpsDescription::Ones(desc.to_local(converter))
            }
            NumericOpsDescription::Zeros(desc) => {
                NumericOpsDescription::Zeros(desc.to_local(converter))
            }
            NumericOpsDescription::Full(desc) => NumericOpsDescription::Full((
                desc.0.to_local(converter),
                local_elem(converter, &desc.1),
            )),
            NumericOpsDescription::Gather(desc) => {
                NumericOpsDescription::Gather(GatherOpsDescription {
                    tensor: desc.tensor.to_local(converter),
                    dim: desc.dim,
                    indices: desc.indices.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::Scatter(desc) => {
                NumericOpsDescription::Scatter(ScatterOpsDescription {
                    tensor: desc.tensor.to_local(converter),
                    dim: desc.dim,
                    indices: desc.indices.to_local(converter),
                    value: desc.value.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::Select(desc) => {
                NumericOpsDescription::Select(SelectOpsDescription {
                    tensor: desc.tensor.to_local(converter),
                    dim: desc.dim,
                    indices: desc.indices.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::SelectAssign(desc) => {
                NumericOpsDescription::SelectAssign(SelectAssignOpsDescription {
                    tensor: desc.tensor.to_local(converter),
                    dim: desc.dim,
                    indices: desc.indices.to_local(converter),
                    value: desc.value.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::MaskWhere(desc) => {
                NumericOpsDescription::MaskWhere(MaskWhereOpsDescription {
                    tensor: desc.tensor.to_local(converter),
                    mask: desc.mask.to_local(converter),
                    value: desc.value.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::MaskFill(desc) => {
                NumericOpsDescription::MaskFill(MaskFillOpsDescription {
                    tensor: desc.tensor.to_local(converter),
                    mask: desc.mask.to_local(converter),
                    value: local_elem(converter, &desc.value),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::MeanDim(desc) => {
                NumericOpsDescription::MeanDim(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: desc.rhs, // Dim should stay the same.
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::Mean(desc) => NumericOpsDescription::Mean(UnaryOpsDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            NumericOpsDescription::Sum(desc) => NumericOpsDescription::Sum(UnaryOpsDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            NumericOpsDescription::SumDim(desc) => {
                NumericOpsDescription::SumDim(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: desc.rhs, // Dim should stay the same.
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::EqualElem(desc) => {
                NumericOpsDescription::EqualElem(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::Greater(desc) => {
                NumericOpsDescription::Greater(BinaryOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: desc.rhs.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::GreaterElem(desc) => {
                NumericOpsDescription::GreaterElem(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::GreaterEqual(desc) => {
                NumericOpsDescription::GreaterEqual(BinaryOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: desc.rhs.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::GreaterEqualElem(desc) => {
                NumericOpsDescription::GreaterEqualElem(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::Lower(desc) => {
                NumericOpsDescription::Lower(BinaryOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: desc.rhs.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::LowerElem(desc) => {
                NumericOpsDescription::LowerElem(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::LowerEqual(desc) => {
                NumericOpsDescription::LowerEqual(BinaryOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: desc.rhs.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::LowerEqualElem(desc) => {
                NumericOpsDescription::LowerEqualElem(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::ArgMax(desc) => {
                NumericOpsDescription::ArgMax(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::ArgMin(desc) => {
                NumericOpsDescription::ArgMin(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::Max(desc) => NumericOpsDescription::Max(UnaryOpsDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            NumericOpsDescription::MaxDimWithIndices(desc) => {
                NumericOpsDescription::MaxDimWithIndices(ReduceDimWithIndicesDescription {
                    tensor: desc.tensor.to_local(converter),
                    dim: desc.dim,
                    out: desc.out.to_local(converter),
                    out_indices: desc.out_indices.to_local(converter),
                })
            }
            NumericOpsDescription::MinDimWithIndices(desc) => {
                NumericOpsDescription::MinDimWithIndices(ReduceDimWithIndicesDescription {
                    tensor: desc.tensor.to_local(converter),
                    dim: desc.dim,
                    out: desc.out.to_local(converter),
                    out_indices: desc.out_indices.to_local(converter),
                })
            }
            NumericOpsDescription::Min(desc) => NumericOpsDescription::Min(UnaryOpsDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            NumericOpsDescription::MaxDim(desc) => {
                NumericOpsDescription::MaxDim(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::MinDim(desc) => {
                NumericOpsDescription::MinDim(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::Clamp(desc) => {
                NumericOpsDescription::Clamp(ClampOpsDescription {
                    tensor: desc.tensor.to_local(converter),
                    min: desc.min,
                    max: desc.max,
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::ClampMax(desc) => {
                NumericOpsDescription::ClampMax(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_local(converter),
                })
            }
            NumericOpsDescription::ClampMin(desc) => {
                NumericOpsDescription::ClampMin(ScalarOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_local(converter),
                })
            }
        }
    }
}

impl BaseOpsDescription {
    pub(crate) fn to_local(&self, converter: &mut LocalGraphConverter) -> Self {
        match self {
            BaseOpsDescription::ToDevice(desc) => {
                BaseOpsDescription::ToDevice(desc.to_local(converter))
            }
            BaseOpsDescription::Reshape(desc) => BaseOpsDescription::Reshape(ReshapeDescription {
                input: desc.input.to_local(converter),
                out: desc.out.to_local(converter),
            }),
            BaseOpsDescription::SwapDims(desc) => {
                BaseOpsDescription::SwapDims(SwapDimsDescription {
                    input: desc.input.to_local(converter),
                    out: desc.out.to_local(converter),
                    dim1: desc.dim1,
                    dim2: desc.dim2,
                })
            }
            BaseOpsDescription::Slice(desc) => BaseOpsDescription::Slice(SliceOpsDescription {
                tensor: desc.tensor.to_local(converter),
                ranges: desc.ranges.clone(),
                out: desc.out.to_local(converter),
            }),
            BaseOpsDescription::SliceAssign(desc) => {
                BaseOpsDescription::SliceAssign(super::SliceAssignOpsDescription {
                    tensor: desc.tensor.to_local(converter),
                    ranges: desc.ranges.clone(),
                    value: desc.value.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            BaseOpsDescription::Equal(desc) => {
                BaseOpsDescription::Equal(super::BinaryOpsDescription {
                    lhs: desc.lhs.to_local(converter),
                    rhs: desc.rhs.to_local(converter),
                    out: desc.out.to_local(converter),
                })
            }
            BaseOpsDescription::Repeat(desc) => {
                BaseOpsDescription::Repeat(super::RepeatOpsDescription {
                    tensor: desc.tensor.to_local(converter),
                    dim: desc.dim,
                    times: desc.times,
                    out: desc.out.to_local(converter),
                })
            }
            BaseOpsDescription::Cat(desc) => BaseOpsDescription::Cat(super::CatOpsDescription {
                tensors: desc
                    .tensors
                    .iter()
                    .map(|tensor| tensor.to_local(converter))
                    .collect(),
                dim: desc.dim,
                out: desc.out.to_local(converter),
            }),
        }
    }
}

impl TensorDescription {
    pub(crate) fn to_local(&self, converter: &mut LocalGraphConverter) -> Self {
        let local_id = if let Some(value) = converter.tensors_global2local.get(&self.id) {
            value.id.clone()
        } else {
            TensorId::new(converter.tensors_local2global.len() as u64)
        };

        let mut local_shape = Vec::with_capacity(self.shape.len());

        for dim in self.shape.iter() {
            if let Some(dim) = converter.shapes_local2global.get(dim) {
                local_shape.push(*dim);
            } else {
                let dim_new = converter.shapes_local2global.len();
                local_shape.push(dim_new);
                converter.shapes_local2global.insert(*dim, dim_new);
            }
        }

        let local_tensor = TensorDescription {
            id: local_id.clone(),
            shape: local_shape,
            status: self.status.clone(),
        };

        converter
            .tensors_local2global
            .insert(local_id, self.clone());
        converter
            .tensors_global2local
            .insert(self.id.clone(), local_tensor.clone());

        local_tensor
    }
}

#[cfg(test)]
mod tests {
    use crate::TensorStatus;

    use super::*;

    #[test]
    fn tensor_description_to_local() {
        let tensor1 = TensorDescription {
            id: TensorId::new(500),
            shape: vec![512, 32, 2048],
            status: TensorStatus::ReadOnly,
        };
        let tensor2 = TensorDescription {
            id: TensorId::new(501),
            shape: vec![512, 128, 2048],
            status: TensorStatus::ReadOnly,
        };
        let mut converter = LocalGraphConverter::default();
        let tensor1_local = tensor1.to_local(&mut converter);
        let tensor2_local = tensor2.to_local(&mut converter);

        assert_eq!(
            tensor1_local,
            TensorDescription {
                id: TensorId::new(0),
                shape: vec![0, 1, 2],
                status: TensorStatus::ReadOnly
            }
        );
        assert_eq!(
            tensor2_local,
            TensorDescription {
                id: TensorId::new(1),
                shape: vec![0, 3, 2],
                status: TensorStatus::ReadOnly
            }
        );
    }
}
