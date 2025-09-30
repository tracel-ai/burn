use burn_ir::*;
use hashbrown::HashMap;

/// The context contains the relative graph tensor mapping so that a relative tensor id can be
/// mapped to an existing tensor that can be fetched and updated with the
/// [handle container](HandleContainer).
///
/// It also contains all scalar values, which can change even for the same graph. They are sorted
/// in the order in which they appear in the graph.
#[allow(clippy::too_many_arguments)]
#[derive(new)]
pub struct Context<'a, H> {
    /// The tensor mapping where local tensor id points to the updated tensor representation.
    pub tensors: &'a mut HashMap<TensorId, TensorIr>,
    /// Handle container to retrieve tensors based on their representation.
    pub handles: &'a mut HandleContainer<H>,
    /// Scalars found in the graph in the order they appeared.
    pub scalars: &'a mut HashMap<ScalarId, ScalarIr>,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
/// Scalar unique identifier.
pub struct ScalarId {
    /// The value.
    pub value: u64,
}

pub(crate) struct OperationConverter {
    tensors_relative2global: HashMap<TensorId, TensorIr>,
    tensors_global2relative: HashMap<TensorId, TensorIr>,
    shapes_global2relative: HashMap<usize, usize>,
    scalars: HashMap<ScalarId, ScalarIr>,
}

impl Default for OperationConverter {
    fn default() -> Self {
        let mut val = Self {
            tensors_relative2global: Default::default(),
            tensors_global2relative: Default::default(),
            shapes_global2relative: Default::default(),
            scalars: Default::default(),
        };

        // global 1 is always shape id 0.
        val.shapes_global2relative.insert(1, 0);

        val
    }
}

/// Fork of a [context](Context) which owns its data.
pub struct ContextOwned<H> {
    tensors: HashMap<TensorId, TensorIr>,
    handles: HandleContainer<H>,
    scalars: HashMap<ScalarId, ScalarIr>,
}

impl<H: Clone> ContextOwned<H> {
    /// Convert into [context](Context).
    pub fn as_context(&mut self) -> Context<'_, H> {
        Context {
            tensors: &mut self.tensors,
            handles: &mut self.handles,
            scalars: &mut self.scalars,
        }
    }

    /// Fork the context again.
    pub fn fork(&self) -> ContextOwned<H> {
        ContextOwned {
            tensors: self.tensors.clone(),
            handles: self.handles.fork(),
            scalars: self.scalars.clone(),
        }
    }
}

impl<H: Clone> Context<'_, H> {
    /// Fork the context into an [owned context](ContextOwned).
    pub fn fork(&self) -> ContextOwned<H> {
        ContextOwned {
            tensors: self.tensors.clone(),
            handles: self.handles.fork(),
            scalars: self.scalars.clone(),
        }
    }
}

pub(crate) trait RelativeOps {
    /// Convert (usually an [`OperationIr`]) to a relative form.
    ///
    /// The id and the shape of tensors will be computed relative to existing
    /// operations in the queue. We do this because we want to fuse operations
    /// that have similar shapes, but we do not care about the exact values.
    ///
    /// Similar we do not care about the exact ids of the tensor, but about their
    /// relative ids (how close they are in the operation queue)
    fn to_relative(&self, converter: &mut OperationConverter) -> Self;
}

impl OperationConverter {
    pub(crate) fn context<'a, H>(
        &'a mut self,
        handles: &'a mut HandleContainer<H>,
    ) -> Context<'a, H> {
        Context {
            handles,
            tensors: &mut self.tensors_relative2global,
            scalars: &mut self.scalars,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.tensors_relative2global.clear();
        self.tensors_global2relative.clear();

        self.shapes_global2relative.clear();
        // global 1 is always shape id 0.
        self.shapes_global2relative.insert(1, 0);

        self.scalars.clear();
    }
}

impl RelativeOps for OperationIr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            OperationIr::BaseFloat(ops) => OperationIr::BaseFloat(ops.to_relative(converter)),
            OperationIr::BaseInt(ops) => OperationIr::BaseInt(ops.to_relative(converter)),
            OperationIr::BaseBool(ops) => OperationIr::BaseBool(ops.to_relative(converter)),
            OperationIr::NumericFloat(dtype, ops) => {
                OperationIr::NumericFloat(*dtype, ops.to_relative(converter))
            }
            OperationIr::NumericInt(dtype, ops) => {
                OperationIr::NumericInt(*dtype, ops.to_relative(converter))
            }
            OperationIr::Bool(ops) => OperationIr::Bool(ops.to_relative(converter)),
            OperationIr::Int(ops) => OperationIr::Int(ops.to_relative(converter)),
            OperationIr::Float(dtype, ops) => {
                OperationIr::Float(*dtype, ops.to_relative(converter))
            }
            OperationIr::Module(ops) => OperationIr::Module(ops.to_relative(converter)),
            OperationIr::Custom(ops) => OperationIr::Custom(ops.to_relative(converter)),
            OperationIr::Init(ops) => OperationIr::Init(ops.to_relative(converter)),
            OperationIr::Drop(tensor) => OperationIr::Drop(tensor.to_relative(converter)),
        }
    }
}

impl RelativeOps for ModuleOperationIr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            ModuleOperationIr::Embedding(desc) => ModuleOperationIr::Embedding(EmbeddingOpIr {
                weights: desc.weights.to_relative(converter),
                indices: desc.indices.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            ModuleOperationIr::EmbeddingBackward(desc) => {
                ModuleOperationIr::EmbeddingBackward(EmbeddingBackwardOpIr {
                    weights: desc.weights.to_relative(converter),
                    out_grad: desc.out_grad.to_relative(converter),
                    indices: desc.indices.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::Conv1d(desc) => ModuleOperationIr::Conv1d(Conv1dOpIr {
                x: desc.x.to_relative(converter),
                weight: desc.weight.to_relative(converter),
                bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                options: desc.options.clone(),
                out: desc.out.to_relative(converter),
            }),
            ModuleOperationIr::Conv2d(desc) => ModuleOperationIr::Conv2d(Conv2dOpIr {
                x: desc.x.to_relative(converter),
                weight: desc.weight.to_relative(converter),
                bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                options: desc.options.clone(),
                out: desc.out.to_relative(converter),
            }),
            ModuleOperationIr::Conv3d(desc) => ModuleOperationIr::Conv3d(Conv3dOpIr {
                x: desc.x.to_relative(converter),
                weight: desc.weight.to_relative(converter),
                bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                options: desc.options.clone(),
                out: desc.out.to_relative(converter),
            }),
            ModuleOperationIr::DeformableConv2d(desc) => {
                ModuleOperationIr::DeformableConv2d(Box::new(DeformConv2dOpIr {
                    x: desc.x.to_relative(converter),
                    offset: desc.offset.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    mask: desc.mask.as_ref().map(|t| t.to_relative(converter)),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                }))
            }
            ModuleOperationIr::DeformableConv2dBackward(desc) => {
                ModuleOperationIr::DeformableConv2dBackward(Box::new(DeformConv2dBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    offset: desc.offset.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    mask: desc.mask.as_ref().map(|t| t.to_relative(converter)),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    out_grad: desc.out_grad.to_relative(converter),
                    options: desc.options.clone(),
                    input_grad: desc.input_grad.to_relative(converter),
                    offset_grad: desc.offset_grad.to_relative(converter),
                    weight_grad: desc.weight_grad.to_relative(converter),
                    mask_grad: desc.mask_grad.as_ref().map(|t| t.to_relative(converter)),
                    bias_grad: desc.bias_grad.as_ref().map(|t| t.to_relative(converter)),
                }))
            }
            ModuleOperationIr::ConvTranspose1d(desc) => {
                ModuleOperationIr::ConvTranspose1d(ConvTranspose1dOpIr {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::ConvTranspose2d(desc) => {
                ModuleOperationIr::ConvTranspose2d(ConvTranspose2dOpIr {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::ConvTranspose3d(desc) => {
                ModuleOperationIr::ConvTranspose3d(ConvTranspose3dOpIr {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::AvgPool1d(desc) => ModuleOperationIr::AvgPool1d(AvgPool1dOpIr {
                x: desc.x.to_relative(converter),
                kernel_size: desc.kernel_size,
                stride: desc.stride,
                padding: desc.padding,
                count_include_pad: desc.count_include_pad,
                out: desc.out.to_relative(converter),
            }),
            ModuleOperationIr::AvgPool2d(desc) => ModuleOperationIr::AvgPool2d(AvgPool2dOpIr {
                x: desc.x.to_relative(converter),
                kernel_size: desc.kernel_size,
                stride: desc.stride,
                padding: desc.padding,
                count_include_pad: desc.count_include_pad,
                out: desc.out.to_relative(converter),
            }),
            ModuleOperationIr::AvgPool1dBackward(desc) => {
                ModuleOperationIr::AvgPool1dBackward(AvgPool1dBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    grad: desc.grad.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::AvgPool2dBackward(desc) => {
                ModuleOperationIr::AvgPool2dBackward(AvgPool2dBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    grad: desc.grad.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::AdaptiveAvgPool1d(desc) => {
                ModuleOperationIr::AdaptiveAvgPool1d(AdaptiveAvgPool1dOpIr {
                    x: desc.x.to_relative(converter),
                    output_size: desc.output_size,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::AdaptiveAvgPool2d(desc) => {
                ModuleOperationIr::AdaptiveAvgPool2d(AdaptiveAvgPool2dOpIr {
                    x: desc.x.to_relative(converter),
                    output_size: desc.output_size,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::AdaptiveAvgPool1dBackward(desc) => {
                ModuleOperationIr::AdaptiveAvgPool1dBackward(AdaptiveAvgPool1dBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    grad: desc.grad.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::AdaptiveAvgPool2dBackward(desc) => {
                ModuleOperationIr::AdaptiveAvgPool2dBackward(AdaptiveAvgPool2dBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    grad: desc.grad.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::MaxPool1d(desc) => ModuleOperationIr::MaxPool1d(MaxPool1dOpIr {
                x: desc.x.to_relative(converter),
                kernel_size: desc.kernel_size,
                stride: desc.stride,
                padding: desc.padding,
                dilation: desc.dilation,
                out: desc.out.to_relative(converter),
            }),
            ModuleOperationIr::MaxPool1dWithIndices(desc) => {
                ModuleOperationIr::MaxPool1dWithIndices(MaxPool1dWithIndicesOpIr {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_relative(converter),
                    out_indices: desc.out_indices.to_relative(converter),
                })
            }
            ModuleOperationIr::MaxPool1dWithIndicesBackward(desc) => {
                ModuleOperationIr::MaxPool1dWithIndicesBackward(MaxPool1dWithIndicesBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    grad: desc.grad.to_relative(converter),
                    indices: desc.indices.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::MaxPool2d(desc) => ModuleOperationIr::MaxPool2d(MaxPool2dOpIr {
                x: desc.x.to_relative(converter),
                kernel_size: desc.kernel_size,
                stride: desc.stride,
                padding: desc.padding,
                dilation: desc.dilation,
                out: desc.out.to_relative(converter),
            }),
            ModuleOperationIr::MaxPool2dWithIndices(desc) => {
                ModuleOperationIr::MaxPool2dWithIndices(MaxPool2dWithIndicesOpIr {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_relative(converter),
                    out_indices: desc.out_indices.to_relative(converter),
                })
            }
            ModuleOperationIr::MaxPool2dWithIndicesBackward(desc) => {
                ModuleOperationIr::MaxPool2dWithIndicesBackward(MaxPool2dWithIndicesBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    grad: desc.grad.to_relative(converter),
                    indices: desc.indices.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::Interpolate(desc) => {
                ModuleOperationIr::Interpolate(InterpolateOpIr {
                    x: desc.x.to_relative(converter),
                    output_size: desc.output_size,
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::InterpolateBackward(desc) => {
                ModuleOperationIr::InterpolateBackward(InterpolateBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    grad: desc.grad.to_relative(converter),
                    output_size: desc.output_size,
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
        }
    }
}

impl RelativeOps for FloatOperationIr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            FloatOperationIr::Exp(desc) => FloatOperationIr::Exp(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Log(desc) => FloatOperationIr::Log(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Log1p(desc) => FloatOperationIr::Log1p(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Erf(desc) => FloatOperationIr::Erf(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::PowfScalar(desc) => FloatOperationIr::PowfScalar(ScalarOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Sqrt(desc) => FloatOperationIr::Sqrt(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Cos(desc) => FloatOperationIr::Cos(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Sin(desc) => FloatOperationIr::Sin(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Tanh(desc) => FloatOperationIr::Tanh(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::IntoInt(desc) => FloatOperationIr::IntoInt(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Matmul(desc) => FloatOperationIr::Matmul(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Cross(desc) => FloatOperationIr::Cross(CrossOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
                dim: desc.dim,
            }),
            FloatOperationIr::Random(desc) => FloatOperationIr::Random(RandomOpIr {
                out: desc.out.to_relative(converter),
                distribution: desc.distribution,
            }),
            FloatOperationIr::Recip(desc) => FloatOperationIr::Recip(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Quantize(desc) => FloatOperationIr::Quantize(QuantizeOpIr {
                tensor: desc.tensor.to_relative(converter),
                qparams: QuantizationParametersIr {
                    scales: desc.qparams.scales.to_relative(converter),
                },
                scheme: desc.scheme,
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Dequantize(desc) => FloatOperationIr::Dequantize(DequantizeOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Round(desc) => FloatOperationIr::Round(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Floor(desc) => FloatOperationIr::Floor(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Ceil(desc) => FloatOperationIr::Ceil(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::IsNan(desc) => FloatOperationIr::IsNan(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::IsInf(desc) => FloatOperationIr::IsInf(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
        }
    }
}

impl RelativeOps for BoolOperationIr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            BoolOperationIr::Ones(desc) => BoolOperationIr::Ones(desc.to_relative(converter)),
            BoolOperationIr::Zeros(desc) => BoolOperationIr::Zeros(desc.to_relative(converter)),
            BoolOperationIr::IntoFloat(desc) => BoolOperationIr::IntoFloat(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BoolOperationIr::IntoInt(desc) => BoolOperationIr::IntoInt(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BoolOperationIr::Not(desc) => BoolOperationIr::Not(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BoolOperationIr::And(desc) => BoolOperationIr::And(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BoolOperationIr::Or(desc) => BoolOperationIr::Or(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
        }
    }
}

impl RelativeOps for IntOperationIr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            IntOperationIr::IntoFloat(desc) => IntOperationIr::IntoFloat(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            IntOperationIr::Matmul(desc) => IntOperationIr::Matmul(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            IntOperationIr::BitwiseAnd(desc) => IntOperationIr::BitwiseAnd(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            IntOperationIr::BitwiseAndScalar(desc) => {
                IntOperationIr::BitwiseAndScalar(ScalarOpIr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationIr::BitwiseOr(desc) => IntOperationIr::BitwiseOr(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            IntOperationIr::BitwiseOrScalar(desc) => IntOperationIr::BitwiseOrScalar(ScalarOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs,
                out: desc.out.to_relative(converter),
            }),
            IntOperationIr::BitwiseXor(desc) => IntOperationIr::BitwiseXor(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            IntOperationIr::BitwiseXorScalar(desc) => {
                IntOperationIr::BitwiseXorScalar(ScalarOpIr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationIr::BitwiseNot(desc) => IntOperationIr::BitwiseNot(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            IntOperationIr::BitwiseLeftShift(desc) => {
                IntOperationIr::BitwiseLeftShift(BinaryOpIr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationIr::BitwiseLeftShiftScalar(desc) => {
                IntOperationIr::BitwiseLeftShiftScalar(ScalarOpIr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationIr::BitwiseRightShift(desc) => {
                IntOperationIr::BitwiseRightShift(BinaryOpIr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationIr::BitwiseRightShiftScalar(desc) => {
                IntOperationIr::BitwiseRightShiftScalar(ScalarOpIr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
        }
    }
}

impl RelativeOps for CustomOpIr {
    fn to_relative(&self, converter: &mut OperationConverter) -> CustomOpIr {
        let id = self.id.clone();

        CustomOpIr {
            id,
            inputs: self
                .inputs
                .iter()
                .map(|x| x.to_relative(converter))
                .collect(),
            outputs: self
                .outputs
                .iter()
                .map(|x| x.to_relative(converter))
                .collect(),
        }
    }
}

impl RelativeOps for NumericOperationIr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            NumericOperationIr::Add(desc) => NumericOperationIr::Add(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::AddScalar(desc) => NumericOperationIr::AddScalar(ScalarOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::Sub(desc) => NumericOperationIr::Sub(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::SubScalar(desc) => NumericOperationIr::SubScalar(ScalarOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::Div(desc) => NumericOperationIr::Div(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::DivScalar(desc) => NumericOperationIr::DivScalar(ScalarOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::Rem(desc) => NumericOperationIr::Rem(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::RemScalar(desc) => NumericOperationIr::RemScalar(ScalarOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::Mul(desc) => NumericOperationIr::Mul(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::MulScalar(desc) => NumericOperationIr::MulScalar(ScalarOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::Abs(desc) => NumericOperationIr::Abs(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::Ones(desc) => NumericOperationIr::Ones(desc.to_relative(converter)),
            NumericOperationIr::Zeros(desc) => {
                NumericOperationIr::Zeros(desc.to_relative(converter))
            }
            NumericOperationIr::Full(desc) => NumericOperationIr::Full((
                desc.0.to_relative(converter),
                desc.1.to_relative(converter),
            )),
            NumericOperationIr::Gather(desc) => NumericOperationIr::Gather(GatherOpIr {
                tensor: desc.tensor.to_relative(converter),
                dim: desc.dim,
                indices: desc.indices.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::Scatter(desc) => NumericOperationIr::Scatter(ScatterOpIr {
                tensor: desc.tensor.to_relative(converter),
                dim: desc.dim,
                indices: desc.indices.to_relative(converter),
                value: desc.value.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::Select(desc) => NumericOperationIr::Select(SelectOpIr {
                tensor: desc.tensor.to_relative(converter),
                dim: desc.dim,
                indices: desc.indices.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::SelectAssign(desc) => {
                NumericOperationIr::SelectAssign(SelectAssignOpIr {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    indices: desc.indices.to_relative(converter),
                    value: desc.value.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationIr::MaskWhere(desc) => NumericOperationIr::MaskWhere(MaskWhereOpIr {
                tensor: desc.tensor.to_relative(converter),
                mask: desc.mask.to_relative(converter),
                value: desc.value.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::MaskFill(desc) => NumericOperationIr::MaskFill(MaskFillOpIr {
                tensor: desc.tensor.to_relative(converter),
                mask: desc.mask.to_relative(converter),
                value: desc.value.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::MeanDim(desc) => NumericOperationIr::MeanDim(ReduceDimOpIr {
                input: desc.input.to_relative(converter),
                axis: desc.axis,
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::Mean(desc) => NumericOperationIr::Mean(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::Sum(desc) => NumericOperationIr::Sum(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::SumDim(desc) => {
                NumericOperationIr::SumDim(ReduceDimOpIr {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                    axis: desc.axis, // Axis should stay the same.
                })
            }
            NumericOperationIr::Prod(desc) => NumericOperationIr::Prod(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::ProdDim(desc) => NumericOperationIr::ProdDim(ReduceDimOpIr {
                input: desc.input.to_relative(converter),
                axis: desc.axis,
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::EqualElem(desc) => NumericOperationIr::EqualElem(ScalarOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::Greater(desc) => NumericOperationIr::Greater(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::GreaterElem(desc) => NumericOperationIr::GreaterElem(ScalarOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::GreaterEqual(desc) => {
                NumericOperationIr::GreaterEqual(BinaryOpIr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationIr::GreaterEqualElem(desc) => {
                NumericOperationIr::GreaterEqualElem(ScalarOpIr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationIr::Lower(desc) => NumericOperationIr::Lower(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::LowerElem(desc) => NumericOperationIr::LowerElem(ScalarOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::LowerEqual(desc) => NumericOperationIr::LowerEqual(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::LowerEqualElem(desc) => {
                NumericOperationIr::LowerEqualElem(ScalarOpIr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationIr::ArgMax(desc) => NumericOperationIr::ArgMax(ReduceDimOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
                axis: desc.axis, // Axis should stay the same.
            }),
            NumericOperationIr::ArgMin(desc) => NumericOperationIr::ArgMin(ReduceDimOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
                axis: desc.axis, // Axis should stay the same.
            }),
            NumericOperationIr::Max(desc) => NumericOperationIr::Max(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::MaxDimWithIndices(desc) => {
                NumericOperationIr::MaxDimWithIndices(ReduceDimWithIndicesOpIr {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    out: desc.out.to_relative(converter),
                    out_indices: desc.out_indices.to_relative(converter),
                })
            }
            NumericOperationIr::MinDimWithIndices(desc) => {
                NumericOperationIr::MinDimWithIndices(ReduceDimWithIndicesOpIr {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    out: desc.out.to_relative(converter),
                    out_indices: desc.out_indices.to_relative(converter),
                })
            }
            NumericOperationIr::Min(desc) => NumericOperationIr::Min(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::MaxDim(desc) => NumericOperationIr::MaxDim(ReduceDimOpIr {
                input: desc.input.to_relative(converter),
                axis: desc.axis,
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::MinDim(desc) => NumericOperationIr::MinDim(ReduceDimOpIr {
                input: desc.input.to_relative(converter),
                axis: desc.axis,
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::MaxAbs(desc) => NumericOperationIr::MaxAbs(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::MaxAbsDim(desc) => NumericOperationIr::MaxAbsDim(ReduceDimOpIr {
                input: desc.input.to_relative(converter),
                axis: desc.axis,
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::Clamp(desc) => NumericOperationIr::Clamp(ClampOpIr {
                tensor: desc.tensor.to_relative(converter),
                min: desc.min.to_relative(converter),
                max: desc.max.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::IntRandom(desc) => NumericOperationIr::IntRandom(RandomOpIr {
                out: desc.out.to_relative(converter),
                distribution: desc.distribution,
            }),
            NumericOperationIr::Powf(desc) => NumericOperationIr::Powf(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
        }
    }
}

impl RelativeOps for BaseOperationIr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            BaseOperationIr::ToDevice(desc) => {
                BaseOperationIr::ToDevice(desc.to_relative(converter))
            }
            BaseOperationIr::Reshape(desc) => BaseOperationIr::Reshape(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationIr::SwapDims(desc) => BaseOperationIr::SwapDims(SwapDimsOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
                dim1: desc.dim1,
                dim2: desc.dim2,
            }),
            BaseOperationIr::Permute(desc) => BaseOperationIr::Permute(PermuteOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
                axes: desc.axes.clone(),
            }),
            BaseOperationIr::Expand(desc) => BaseOperationIr::Expand(ExpandOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
                shape: desc.shape.clone(),
            }),
            BaseOperationIr::Unfold(desc) => BaseOperationIr::Unfold(UnfoldOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
                dim: desc.dim,
                size: desc.size,
                step: desc.step,
            }),
            BaseOperationIr::Flip(desc) => BaseOperationIr::Flip(FlipOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
                axes: desc.axes.clone(),
            }),
            BaseOperationIr::Slice(desc) => BaseOperationIr::Slice(SliceOpIr {
                tensor: desc.tensor.to_relative(converter),
                ranges: desc
                    .ranges
                    .iter()
                    .map(|_info| burn_tensor::Slice::from(0..1))
                    .collect(),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationIr::SliceAssign(desc) => BaseOperationIr::SliceAssign(SliceAssignOpIr {
                tensor: desc.tensor.to_relative(converter),
                ranges: desc
                    .ranges
                    .iter()
                    .map(|_range| burn_tensor::Slice::from(0..1))
                    .collect(),
                value: desc.value.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationIr::Equal(desc) => BaseOperationIr::Equal(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationIr::RepeatDim(desc) => BaseOperationIr::RepeatDim(RepeatDimOpIr {
                tensor: desc.tensor.to_relative(converter),
                dim: desc.dim,
                times: desc.times,
                out: desc.out.to_relative(converter),
            }),
            BaseOperationIr::Cat(desc) => BaseOperationIr::Cat(CatOpIr {
                tensors: desc
                    .tensors
                    .iter()
                    .map(|tensor| tensor.to_relative(converter))
                    .collect(),
                dim: desc.dim,
                out: desc.out.to_relative(converter),
            }),
            BaseOperationIr::Cast(desc) => BaseOperationIr::Cast(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationIr::Empty(desc) => BaseOperationIr::Empty(desc.to_relative(converter)),
        }
    }
}

impl RelativeOps for InitOperationIr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        Self {
            out: self.out.to_relative(converter),
        }
    }
}

impl RelativeOps for TensorIr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        let relative_id = self.id.to_relative(converter);

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
        let relative_tensor = TensorIr {
            id: relative_id,
            shape: relative_shape,
            status: self.status,
            dtype: self.dtype,
        };

        // We update both mappings.
        converter
            .tensors_relative2global
            .insert(relative_id, self.clone());
        converter
            .tensors_global2relative
            .insert(self.id, relative_tensor.clone());

        relative_tensor
    }
}

impl RelativeOps for TensorId {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        if let Some(value) = converter.tensors_global2relative.get(self) {
            // If we already have the same tensor registered, we have to update its value, but not
            // its id.
            value.id
        } else {
            // We create a new relative id since we never seen this tensor in the graph before.
            TensorId::new(converter.tensors_relative2global.len() as u64)
        }
    }
}

impl RelativeOps for ScalarIr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        if matches!(self, ScalarIr::Bool(_)) {
            todo!("Unsupported dtype ({self:?}) for scalar")
        }

        let id = ScalarId {
            value: converter.scalars.len() as u64,
        };

        converter.scalars.insert(id, *self);
        ScalarIr::U64(id.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ir::{TensorId, TensorIr, TensorStatus};
    use burn_tensor::DType;

    #[test]
    fn tensor_description_to_relative() {
        let tensor1 = TensorIr {
            id: TensorId::new(500),
            shape: vec![512, 32, 2048],
            status: TensorStatus::ReadOnly,
            dtype: DType::F32,
        };
        let tensor2 = TensorIr {
            id: TensorId::new(501),
            shape: vec![512, 128, 2048],
            status: TensorStatus::ReadOnly,
            dtype: DType::F32,
        };
        let mut converter = OperationConverter::default();
        let tensor1_local = tensor1.to_relative(&mut converter);
        let tensor2_local = tensor2.to_relative(&mut converter);

        assert_eq!(
            tensor1_local,
            TensorIr {
                id: TensorId::new(0),
                shape: vec![1, 2, 3],
                status: TensorStatus::ReadOnly,
                dtype: DType::F32
            }
        );
        assert_eq!(
            tensor2_local,
            TensorIr {
                id: TensorId::new(1),
                shape: vec![1, 4, 3],
                status: TensorStatus::ReadOnly,
                dtype: DType::F32
            }
        );
    }

    #[test]
    fn scalar_ir_to_relative() {
        let scalar1 = ScalarIr::F32(1.0);
        let scalar2 = ScalarIr::U8(1);
        let mut converter = OperationConverter::default();
        let scalar1_local = scalar1.to_relative(&mut converter);
        let scalar2_local = scalar2.to_relative(&mut converter);

        assert_eq!(scalar1_local, ScalarIr::U64(0));
        assert_eq!(scalar2_local, ScalarIr::U64(1));
    }
}
