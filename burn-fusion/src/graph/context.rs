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

/// The context contains the relative graph tensor mapping so that a relative tensor id can be
/// mapped to an existing tensor that can be fetched and updated with the
/// [handle container](HandleContainer).
///
/// It also contains all scalar values, which can change even for the same graph. They are sorted
/// in the order in which they appear in the graph.
#[derive(new)]
pub struct Context<'a, B: FusionBackend> {
    /// The tensor mapping where local tensor id points to the updated tensor description.
    pub tensors: &'a HashMap<TensorId, TensorDescription>,
    /// Handle container to retrieve tensors based on their description.
    pub handles: &'a mut HandleContainer<B>,
    /// Float scalars found in the graph in the order they appeared.
    pub scalar_floats: &'a Vec<f32>,
    /// Int scalars found in the graph in the order they appeared.
    pub scalar_ints: &'a Vec<i32>,
}

#[derive(Default)]
pub(crate) struct RelativeGraphConverter {
    tensors_relative2global: HashMap<TensorId, TensorDescription>,
    tensors_global2relative: HashMap<TensorId, TensorDescription>,
    /// Only useful to create new shape ID.
    /// You should use tensor descriptions to retrieve the proper shape.
    shapes_global2relative: HashMap<usize, usize>,
    scalar_floats: Vec<f32>,
    scalar_ints: Vec<i32>,
}

impl RelativeGraphConverter {
    pub(crate) fn context<'a, B: FusionBackend>(
        &'a self,
        handles: &'a mut HandleContainer<B>,
    ) -> Context<'a, B> {
        Context {
            handles,
            tensors: &self.tensors_relative2global,
            scalar_floats: &self.scalar_floats,
            scalar_ints: &self.scalar_ints,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.tensors_relative2global.clear();
        self.tensors_global2relative.clear();
        self.shapes_global2relative.clear();
        self.scalar_floats.clear();
        self.scalar_ints.clear();
    }

    pub(crate) fn relative_float<E: Element>(&mut self, elem: &E) -> E {
        self.scalar_floats.push(elem.elem());
        // We return 0 so that the id from a scalar operation is the same no matter its scalar
        // value.
        0.elem()
    }

    pub(crate) fn relative_int<E: Element>(&mut self, elem: &E) -> E {
        self.scalar_ints.push(elem.elem());
        // We return 0 so that the id from a scalar operation is the same no matter its scalar
        // value.
        0.elem()
    }
}

impl TensorOpsDescription {
    pub(crate) fn to_relative(&self, converter: &mut RelativeGraphConverter) -> Self {
        match self {
            TensorOpsDescription::BaseOpsFloat(ops) => {
                TensorOpsDescription::BaseOpsFloat(ops.to_relative(converter))
            }
            TensorOpsDescription::BaseOpsInt(ops) => {
                TensorOpsDescription::BaseOpsInt(ops.to_relative(converter))
            }
            TensorOpsDescription::BaseOpsBool(ops) => {
                TensorOpsDescription::BaseOpsBool(ops.to_relative(converter))
            }
            TensorOpsDescription::NumericOpsFloat(ops) => TensorOpsDescription::NumericOpsFloat(
                ops.to_relative(converter, |converter, e| converter.relative_float(e)),
            ),
            TensorOpsDescription::NumericOpsInt(ops) => TensorOpsDescription::NumericOpsInt(
                ops.to_relative(converter, |converter, e| converter.relative_int(e)),
            ),
            TensorOpsDescription::BoolOps(ops) => {
                TensorOpsDescription::BoolOps(ops.to_relative(converter))
            }
            TensorOpsDescription::IntOps(ops) => {
                TensorOpsDescription::IntOps(ops.to_relative(converter))
            }
            TensorOpsDescription::FloatOps(ops) => {
                TensorOpsDescription::FloatOps(ops.to_relative(converter))
            }
            TensorOpsDescription::ModuleOps(ops) => {
                TensorOpsDescription::ModuleOps(ops.to_relative(converter))
            }
        }
    }
}

impl ModuleOpsDescription {
    pub(crate) fn to_relative(&self, converter: &mut RelativeGraphConverter) -> Self {
        match self {
            ModuleOpsDescription::Embedding(desc) => {
                ModuleOpsDescription::Embedding(EmbeddingDescription {
                    weights: desc.weights.to_relative(converter),
                    indices: desc.indices.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOpsDescription::EmbeddingBackward(desc) => {
                ModuleOpsDescription::EmbeddingBackward(EmbeddingBackwardDescription {
                    weights: desc.weights.to_relative(converter),
                    out_grad: desc.out_grad.to_relative(converter),
                    indices: desc.indices.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOpsDescription::Conv1d(desc) => ModuleOpsDescription::Conv1d(Conv1dDescription {
                x: desc.x.to_relative(converter),
                weight: desc.weight.to_relative(converter),
                bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                options: desc.options.clone(),
                out: desc.out.to_relative(converter),
            }),
            ModuleOpsDescription::Conv2d(desc) => ModuleOpsDescription::Conv2d(Conv2dDescription {
                x: desc.x.to_relative(converter),
                weight: desc.weight.to_relative(converter),
                bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                options: desc.options.clone(),
                out: desc.out.to_relative(converter),
            }),
            ModuleOpsDescription::ConvTranspose1d(desc) => {
                ModuleOpsDescription::ConvTranspose1d(ConvTranspose1dDescription {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOpsDescription::ConvTranspose2d(desc) => {
                ModuleOpsDescription::ConvTranspose2d(ConvTranspose2dDescription {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOpsDescription::AvgPool1d(desc) => {
                ModuleOpsDescription::AvgPool1d(super::AvgPool1dDescription {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOpsDescription::AvgPool2d(desc) => {
                ModuleOpsDescription::AvgPool2d(AvgPool2dDescription {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOpsDescription::AvgPool1dBackward(desc) => {
                ModuleOpsDescription::AvgPool1dBackward(super::AvgPool1dBackwardDescription {
                    x: desc.x.to_relative(converter),
                    grad: desc.grad.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOpsDescription::AvgPool2dBackward(desc) => {
                ModuleOpsDescription::AvgPool2dBackward(AvgPool2dBackwardDescription {
                    x: desc.x.to_relative(converter),
                    grad: desc.grad.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOpsDescription::AdaptiveAvgPool1d(desc) => {
                ModuleOpsDescription::AdaptiveAvgPool1d(AdaptiveAvgPool1dDescription {
                    x: desc.x.to_relative(converter),
                    output_size: desc.output_size,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOpsDescription::AdaptiveAvgPool2d(desc) => {
                ModuleOpsDescription::AdaptiveAvgPool2d(AdaptiveAvgPool2dDescription {
                    x: desc.x.to_relative(converter),
                    output_size: desc.output_size,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOpsDescription::AdaptiveAvgPool1dBackward(desc) => {
                ModuleOpsDescription::AdaptiveAvgPool1dBackward(
                    AdaptiveAvgPool1dBackwardDescription {
                        x: desc.x.to_relative(converter),
                        grad: desc.grad.to_relative(converter),
                        out: desc.out.to_relative(converter),
                    },
                )
            }
            ModuleOpsDescription::AdaptiveAvgPool2dBackward(desc) => {
                ModuleOpsDescription::AdaptiveAvgPool2dBackward(
                    AdaptiveAvgPool2dBackwardDescription {
                        x: desc.x.to_relative(converter),
                        grad: desc.grad.to_relative(converter),
                        out: desc.out.to_relative(converter),
                    },
                )
            }
            ModuleOpsDescription::MaxPool1d(desc) => {
                ModuleOpsDescription::MaxPool1d(MaxPool1dDescription {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOpsDescription::MaxPool1dWithIndices(desc) => {
                ModuleOpsDescription::MaxPool1dWithIndices(MaxPool1dWithIndicesDescription {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_relative(converter),
                    out_indices: desc.out_indices.to_relative(converter),
                })
            }
            ModuleOpsDescription::MaxPool1dWithIndicesBackward(desc) => {
                ModuleOpsDescription::MaxPool1dWithIndicesBackward(
                    MaxPool1dWithIndicesBackwardDescription {
                        x: desc.x.to_relative(converter),
                        grad: desc.grad.to_relative(converter),
                        indices: desc.indices.to_relative(converter),
                        kernel_size: desc.kernel_size,
                        stride: desc.stride,
                        padding: desc.padding,
                        dilation: desc.dilation,
                        out: desc.out.to_relative(converter),
                    },
                )
            }
            ModuleOpsDescription::MaxPool2d(desc) => {
                ModuleOpsDescription::MaxPool2d(MaxPool2dDescription {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOpsDescription::MaxPool2dWithIndices(desc) => {
                ModuleOpsDescription::MaxPool2dWithIndices(MaxPool2dWithIndicesDescription {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_relative(converter),
                    out_indices: desc.out_indices.to_relative(converter),
                })
            }
            ModuleOpsDescription::MaxPool2dWithIndicesBackward(desc) => {
                ModuleOpsDescription::MaxPool2dWithIndicesBackward(
                    MaxPool2dWithIndicesBackwardDescription {
                        x: desc.x.to_relative(converter),
                        grad: desc.grad.to_relative(converter),
                        indices: desc.indices.to_relative(converter),
                        kernel_size: desc.kernel_size,
                        stride: desc.stride,
                        padding: desc.padding,
                        dilation: desc.dilation,
                        out: desc.out.to_relative(converter),
                    },
                )
            }
        }
    }
}

impl FloatOpsDescription {
    pub(crate) fn to_relative(&self, converter: &mut RelativeGraphConverter) -> Self {
        match self {
            FloatOpsDescription::Exp(desc) => FloatOpsDescription::Exp(UnaryOpsDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOpsDescription::Log(desc) => FloatOpsDescription::Log(UnaryOpsDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOpsDescription::Log1p(desc) => FloatOpsDescription::Log1p(UnaryOpsDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOpsDescription::Erf(desc) => FloatOpsDescription::Erf(UnaryOpsDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOpsDescription::Powf(desc) => FloatOpsDescription::Powf(ScalarOpsDescription {
                lhs: desc.lhs.to_relative(converter),
                rhs: converter.relative_float(&desc.rhs),
                out: desc.out.to_relative(converter),
            }),
            FloatOpsDescription::Sqrt(desc) => FloatOpsDescription::Sqrt(UnaryOpsDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOpsDescription::Cos(desc) => FloatOpsDescription::Cos(UnaryOpsDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOpsDescription::Sin(desc) => FloatOpsDescription::Sin(UnaryOpsDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOpsDescription::Tanh(desc) => FloatOpsDescription::Tanh(UnaryOpsDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOpsDescription::IntoInt(desc) => {
                FloatOpsDescription::IntoInt(UnaryOpsDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOpsDescription::Matmul(desc) => {
                FloatOpsDescription::Matmul(BinaryOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOpsDescription::Random(desc) => {
                FloatOpsDescription::Random(RandomOpsDescription {
                    out: desc.out.to_relative(converter),
                    distribution: desc.distribution,
                })
            }
            FloatOpsDescription::Recip(desc) => FloatOpsDescription::Recip(UnaryOpsDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
        }
    }
}

impl BoolOpsDescription {
    pub(crate) fn to_relative(&self, converter: &mut RelativeGraphConverter) -> Self {
        match self {
            BoolOpsDescription::IntoFloat(desc) => {
                BoolOpsDescription::IntoFloat(UnaryOpsDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            BoolOpsDescription::IntoInt(desc) => BoolOpsDescription::IntoInt(UnaryOpsDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BoolOpsDescription::Not(desc) => BoolOpsDescription::Not(UnaryOpsDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
        }
    }
}

impl IntOpsDescription {
    pub(crate) fn to_relative(&self, converter: &mut RelativeGraphConverter) -> Self {
        match self {
            IntOpsDescription::IntoFloat(desc) => {
                IntOpsDescription::IntoFloat(UnaryOpsDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
        }
    }
}

impl<E: Element> NumericOpsDescription<E> {
    pub(crate) fn to_relative<F>(
        &self,
        converter: &mut RelativeGraphConverter,
        local_elem: F,
    ) -> Self
    where
        F: Fn(&mut RelativeGraphConverter, &E) -> E,
    {
        match self {
            NumericOpsDescription::Add(desc) => NumericOpsDescription::Add(BinaryOpsDescription {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOpsDescription::AddScalar(desc) => {
                NumericOpsDescription::AddScalar(ScalarOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::Sub(desc) => NumericOpsDescription::Sub(BinaryOpsDescription {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOpsDescription::SubScalar(desc) => {
                NumericOpsDescription::SubScalar(ScalarOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::Div(desc) => NumericOpsDescription::Div(BinaryOpsDescription {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOpsDescription::DivScalar(desc) => {
                NumericOpsDescription::DivScalar(ScalarOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::Mul(desc) => NumericOpsDescription::Mul(BinaryOpsDescription {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOpsDescription::MulScalar(desc) => {
                NumericOpsDescription::MulScalar(ScalarOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::Abs(desc) => NumericOpsDescription::Abs(UnaryOpsDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOpsDescription::Ones(desc) => {
                NumericOpsDescription::Ones(desc.to_relative(converter))
            }
            NumericOpsDescription::Zeros(desc) => {
                NumericOpsDescription::Zeros(desc.to_relative(converter))
            }
            NumericOpsDescription::Full(desc) => NumericOpsDescription::Full((
                desc.0.to_relative(converter),
                local_elem(converter, &desc.1),
            )),
            NumericOpsDescription::Gather(desc) => {
                NumericOpsDescription::Gather(GatherOpsDescription {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    indices: desc.indices.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::Scatter(desc) => {
                NumericOpsDescription::Scatter(ScatterOpsDescription {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    indices: desc.indices.to_relative(converter),
                    value: desc.value.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::Select(desc) => {
                NumericOpsDescription::Select(SelectOpsDescription {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    indices: desc.indices.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::SelectAssign(desc) => {
                NumericOpsDescription::SelectAssign(SelectAssignOpsDescription {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    indices: desc.indices.to_relative(converter),
                    value: desc.value.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::MaskWhere(desc) => {
                NumericOpsDescription::MaskWhere(MaskWhereOpsDescription {
                    tensor: desc.tensor.to_relative(converter),
                    mask: desc.mask.to_relative(converter),
                    value: desc.value.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::MaskFill(desc) => {
                NumericOpsDescription::MaskFill(MaskFillOpsDescription {
                    tensor: desc.tensor.to_relative(converter),
                    mask: desc.mask.to_relative(converter),
                    value: local_elem(converter, &desc.value),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::MeanDim(desc) => {
                NumericOpsDescription::MeanDim(ScalarOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs, // Dim should stay the same.
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::Mean(desc) => NumericOpsDescription::Mean(UnaryOpsDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOpsDescription::Sum(desc) => NumericOpsDescription::Sum(UnaryOpsDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOpsDescription::SumDim(desc) => {
                NumericOpsDescription::SumDim(ScalarOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs, // Dim should stay the same.
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::EqualElem(desc) => {
                NumericOpsDescription::EqualElem(ScalarOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::Greater(desc) => {
                NumericOpsDescription::Greater(BinaryOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::GreaterElem(desc) => {
                NumericOpsDescription::GreaterElem(ScalarOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::GreaterEqual(desc) => {
                NumericOpsDescription::GreaterEqual(BinaryOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::GreaterEqualElem(desc) => {
                NumericOpsDescription::GreaterEqualElem(ScalarOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::Lower(desc) => {
                NumericOpsDescription::Lower(BinaryOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::LowerElem(desc) => {
                NumericOpsDescription::LowerElem(ScalarOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::LowerEqual(desc) => {
                NumericOpsDescription::LowerEqual(BinaryOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::LowerEqualElem(desc) => {
                NumericOpsDescription::LowerEqualElem(ScalarOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::ArgMax(desc) => {
                NumericOpsDescription::ArgMax(ScalarOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::ArgMin(desc) => {
                NumericOpsDescription::ArgMin(ScalarOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::Max(desc) => NumericOpsDescription::Max(UnaryOpsDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOpsDescription::MaxDimWithIndices(desc) => {
                NumericOpsDescription::MaxDimWithIndices(ReduceDimWithIndicesDescription {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    out: desc.out.to_relative(converter),
                    out_indices: desc.out_indices.to_relative(converter),
                })
            }
            NumericOpsDescription::MinDimWithIndices(desc) => {
                NumericOpsDescription::MinDimWithIndices(ReduceDimWithIndicesDescription {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    out: desc.out.to_relative(converter),
                    out_indices: desc.out_indices.to_relative(converter),
                })
            }
            NumericOpsDescription::Min(desc) => NumericOpsDescription::Min(UnaryOpsDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOpsDescription::MaxDim(desc) => {
                NumericOpsDescription::MaxDim(ScalarOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::MinDim(desc) => {
                NumericOpsDescription::MinDim(ScalarOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOpsDescription::Clamp(desc) => {
                NumericOpsDescription::Clamp(ClampOpsDescription {
                    tensor: desc.tensor.to_relative(converter),
                    min: local_elem(converter, &desc.min),
                    max: local_elem(converter, &desc.max),
                    out: desc.out.to_relative(converter),
                })
            }
        }
    }
}

impl BaseOpsDescription {
    pub(crate) fn to_relative(&self, converter: &mut RelativeGraphConverter) -> Self {
        match self {
            BaseOpsDescription::ToDevice(desc) => {
                BaseOpsDescription::ToDevice(desc.to_relative(converter))
            }
            BaseOpsDescription::Reshape(desc) => BaseOpsDescription::Reshape(ReshapeDescription {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BaseOpsDescription::SwapDims(desc) => {
                BaseOpsDescription::SwapDims(SwapDimsDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                    dim1: desc.dim1,
                    dim2: desc.dim2,
                })
            }
            BaseOpsDescription::Slice(desc) => BaseOpsDescription::Slice(SliceOpsDescription {
                tensor: desc.tensor.to_relative(converter),
                ranges: desc.ranges.clone(),
                out: desc.out.to_relative(converter),
            }),
            BaseOpsDescription::SliceAssign(desc) => {
                BaseOpsDescription::SliceAssign(super::SliceAssignOpsDescription {
                    tensor: desc.tensor.to_relative(converter),
                    ranges: desc.ranges.clone(),
                    value: desc.value.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            BaseOpsDescription::Equal(desc) => {
                BaseOpsDescription::Equal(super::BinaryOpsDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            BaseOpsDescription::Repeat(desc) => {
                BaseOpsDescription::Repeat(super::RepeatOpsDescription {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    times: desc.times,
                    out: desc.out.to_relative(converter),
                })
            }
            BaseOpsDescription::Cat(desc) => BaseOpsDescription::Cat(super::CatOpsDescription {
                tensors: desc
                    .tensors
                    .iter()
                    .map(|tensor| tensor.to_relative(converter))
                    .collect(),
                dim: desc.dim,
                out: desc.out.to_relative(converter),
            }),
        }
    }
}

impl TensorDescription {
    pub(crate) fn to_relative(&self, converter: &mut RelativeGraphConverter) -> Self {
        let relative_id = if let Some(value) = converter.tensors_global2relative.get(&self.id) {
            // If we already have the same tensor registered, we have to update its value, but not
            // its id.
            value.id.clone()
        } else {
            // We create a new relative id since we never seen this tensor in the graph before.
            TensorId::new(converter.tensors_relative2global.len() as u64)
        };

        // We can create relative shapes by mapping each shape found to an ID, which is a `usize`.
        let mut relative_shape = Vec::with_capacity(self.shape.len());
        for dim in self.shape.iter() {
            if let Some(dim_id) = converter.shapes_global2relative.get(dim) {
                // We already saw that dim value before, so we retrieve its ID.
                relative_shape.push(*dim_id);
            } else {
                // We never saw this dim value before, therefore we create a new ID.
                let dim_id = converter.shapes_global2relative.len();
                relative_shape.push(dim_id);
                converter.shapes_global2relative.insert(*dim, dim_id);
            }
        }

        // We create the relative tensor.
        let relative_tensor = TensorDescription {
            id: relative_id.clone(),
            shape: relative_shape,
            status: self.status.clone(),
        };

        // We update both mappings.
        converter
            .tensors_relative2global
            .insert(relative_id, self.clone());
        converter
            .tensors_global2relative
            .insert(self.id.clone(), relative_tensor.clone());

        relative_tensor
    }
}

#[cfg(test)]
mod tests {
    use crate::TensorStatus;

    use super::*;

    #[test]
    fn tensor_description_to_relative() {
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
        let mut converter = RelativeGraphConverter::default();
        let tensor1_local = tensor1.to_relative(&mut converter);
        let tensor2_local = tensor2.to_relative(&mut converter);

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
