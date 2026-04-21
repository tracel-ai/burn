use burn_backend::{Shape, Slice};
use burn_ir::*;
use hashbrown::HashMap;

/// The context contains the relative graph tensor mapping so that a relative tensor id can be
/// mapped to an existing tensor that can be fetched and updated with the
/// [handle container](HandleContainer).
///
/// It also contains all scalar values, which can change even for the same graph. They are sorted
/// in the order in which they appear in the graph.
pub struct Context<H> {
    /// The tensor mapping where local tensor id points to the updated tensor representation.
    pub tensors: HashMap<TensorId, TensorIr>,
    /// Handle container to retrieve tensors based on their representation.
    pub handles: HandleContainer<H>,
    /// Scalars found in the graph in the order they appeared.
    pub scalars: HashMap<ScalarId, ScalarIr>,
    /// Shape mapping from relative shape ids to global (real) shape ids.
    pub shapes_relative2global: HashMap<usize, usize>,
}

impl<H: Clone> Context<H> {
    /// Fork the context into an independent owned copy. Used by autotuning to give each
    /// benchmark run a sandbox — mutations stay local until the caller merges new handles
    /// back. `HandleContainer::fork` clones the id→handle map; actual GPU buffers are
    /// refcounted so only the map itself is duplicated.
    pub fn fork(&self) -> Self {
        Self {
            tensors: self.tensors.clone(),
            handles: self.handles.fork(),
            scalars: self.scalars.clone(),
            shapes_relative2global: self.shapes_relative2global.clone(),
        }
    }
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
    shapes_relative2global: HashMap<usize, usize>,
    scalars: HashMap<ScalarId, ScalarIr>,
}

impl Default for OperationConverter {
    fn default() -> Self {
        let mut val = Self {
            tensors_relative2global: Default::default(),
            tensors_global2relative: Default::default(),
            shapes_global2relative: Default::default(),
            shapes_relative2global: Default::default(),
            scalars: Default::default(),
        };

        // global 1 is always shape id 0.
        val.shapes_global2relative.insert(1, 0);
        val.shapes_relative2global.insert(0, 1);

        val
    }
}

/// RAII guard that temporarily lends three [`OperationConverter`] fields plus a
/// [`HandleContainer`] into a single owned [`Context`]. On drop, the guard moves every
/// field back into its original location.
pub(crate) struct ContextGuard<'a, H> {
    context: Option<Context<H>>,
    converter: &'a mut OperationConverter,
    handles: &'a mut HandleContainer<H>,
}

impl<'a, H> ContextGuard<'a, H> {
    /// Move the converter's per-block exposed state and the server's handle container into a
    /// fresh [`Context`]. The originals are left holding `Default` placeholders; the guard
    /// will swap them back on drop.
    pub(crate) fn new(
        converter: &'a mut OperationConverter,
        handles: &'a mut HandleContainer<H>,
    ) -> Self {
        let context = Context {
            tensors: core::mem::take(&mut converter.tensors_relative2global),
            scalars: core::mem::take(&mut converter.scalars),
            shapes_relative2global: core::mem::take(&mut converter.shapes_relative2global),
            handles: core::mem::take(handles),
        };

        Self {
            context: Some(context),
            converter,
            handles,
        }
    }
}

impl<H> core::ops::Deref for ContextGuard<'_, H> {
    type Target = Context<H>;

    fn deref(&self) -> &Self::Target {
        self.context.as_ref().expect("context guard is alive")
    }
}

impl<H> core::ops::DerefMut for ContextGuard<'_, H> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.context.as_mut().expect("context guard is alive")
    }
}

impl<H> Drop for ContextGuard<'_, H> {
    fn drop(&mut self) {
        if let Some(ctx) = self.context.take() {
            self.converter.tensors_relative2global = ctx.tensors;
            self.converter.scalars = ctx.scalars;
            self.converter.shapes_relative2global = ctx.shapes_relative2global;
            *self.handles = ctx.handles;
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
    pub(crate) fn clear(&mut self) {
        self.tensors_relative2global.clear();
        self.tensors_global2relative.clear();

        self.shapes_global2relative.clear();
        self.shapes_relative2global.clear();

        // global 1 is always shape id 0.
        self.shapes_global2relative.insert(1, 0);
        self.shapes_relative2global.insert(0, 1);

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
            #[cfg(feature = "distributed")]
            OperationIr::Distributed(ops) => OperationIr::Distributed(ops.to_relative(converter)),
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
            ModuleOperationIr::Conv1dXBackward(desc) => {
                ModuleOperationIr::Conv1dXBackward(Conv1dXBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    output_grad: desc.output_grad.to_relative(converter),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::Conv1dWeightBackward(desc) => {
                ModuleOperationIr::Conv1dWeightBackward(Conv1dWeightBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    output_grad: desc.output_grad.to_relative(converter),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::Conv1dBiasBackward(desc) => {
                ModuleOperationIr::Conv1dBiasBackward(Conv1dBiasBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    bias: desc.bias.to_relative(converter),
                    output_grad: desc.output_grad.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::Conv2d(desc) => ModuleOperationIr::Conv2d(Conv2dOpIr {
                x: desc.x.to_relative(converter),
                weight: desc.weight.to_relative(converter),
                bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                options: desc.options.clone(),
                out: desc.out.to_relative(converter),
            }),
            ModuleOperationIr::Conv2dXBackward(desc) => {
                ModuleOperationIr::Conv2dXBackward(Conv2dXBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    output_grad: desc.output_grad.to_relative(converter),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::Conv2dWeightBackward(desc) => {
                ModuleOperationIr::Conv2dWeightBackward(Conv2dWeightBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    output_grad: desc.output_grad.to_relative(converter),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::Conv2dBiasBackward(desc) => {
                ModuleOperationIr::Conv2dBiasBackward(Conv2dBiasBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    bias: desc.bias.to_relative(converter),
                    output_grad: desc.output_grad.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::Conv3d(desc) => ModuleOperationIr::Conv3d(Conv3dOpIr {
                x: desc.x.to_relative(converter),
                weight: desc.weight.to_relative(converter),
                bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                options: desc.options.clone(),
                out: desc.out.to_relative(converter),
            }),
            ModuleOperationIr::Conv3dXBackward(desc) => {
                ModuleOperationIr::Conv3dXBackward(Conv3dXBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    output_grad: desc.output_grad.to_relative(converter),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::Conv3dWeightBackward(desc) => {
                ModuleOperationIr::Conv3dWeightBackward(Conv3dWeightBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    output_grad: desc.output_grad.to_relative(converter),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::Conv3dBiasBackward(desc) => {
                ModuleOperationIr::Conv3dBiasBackward(Conv3dBiasBackwardOpIr {
                    x: desc.x.to_relative(converter),
                    bias: desc.bias.to_relative(converter),
                    output_grad: desc.output_grad.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
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
                ceil_mode: desc.ceil_mode,
                out: desc.out.to_relative(converter),
            }),
            ModuleOperationIr::AvgPool2d(desc) => ModuleOperationIr::AvgPool2d(AvgPool2dOpIr {
                x: desc.x.to_relative(converter),
                kernel_size: desc.kernel_size,
                stride: desc.stride,
                padding: desc.padding,
                count_include_pad: desc.count_include_pad,
                ceil_mode: desc.ceil_mode,
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
                    ceil_mode: desc.ceil_mode,
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
                    ceil_mode: desc.ceil_mode,
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
                ceil_mode: desc.ceil_mode,
                out: desc.out.to_relative(converter),
            }),
            ModuleOperationIr::MaxPool1dWithIndices(desc) => {
                ModuleOperationIr::MaxPool1dWithIndices(MaxPool1dWithIndicesOpIr {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    ceil_mode: desc.ceil_mode,
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
                    ceil_mode: desc.ceil_mode,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationIr::MaxPool2d(desc) => ModuleOperationIr::MaxPool2d(MaxPool2dOpIr {
                x: desc.x.to_relative(converter),
                kernel_size: desc.kernel_size,
                stride: desc.stride,
                padding: desc.padding,
                dilation: desc.dilation,
                ceil_mode: desc.ceil_mode,
                out: desc.out.to_relative(converter),
            }),
            ModuleOperationIr::MaxPool2dWithIndices(desc) => {
                ModuleOperationIr::MaxPool2dWithIndices(MaxPool2dWithIndicesOpIr {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    ceil_mode: desc.ceil_mode,
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
                    ceil_mode: desc.ceil_mode,
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
            ModuleOperationIr::Rfft(desc) => ModuleOperationIr::Rfft(RfftOpIr {
                signal: desc.signal.to_relative(converter),
                dim: desc.dim,
                out_re: desc.out_re.to_relative(converter),
                out_im: desc.out_re.to_relative(converter),
            }),
            ModuleOperationIr::IRfft(desc) => ModuleOperationIr::IRfft(IRfftOpIr {
                input_re: desc.input_re.to_relative(converter),
                input_im: desc.input_im.to_relative(converter),
                dim: desc.dim,
                out_signal: desc.out_signal.to_relative(converter),
            }),
            ModuleOperationIr::Attention(desc) => ModuleOperationIr::Attention(AttentionOpIr {
                query: desc.query.to_relative(converter),
                key: desc.key.to_relative(converter),
                value: desc.value.to_relative(converter),
                mask: desc.mask.as_ref().map(|m| m.to_relative(converter)),
                attn_bias: desc.attn_bias.as_ref().map(|ab| ab.to_relative(converter)),
                options: desc.options.clone(),
                out: desc.out.to_relative(converter),
            }),
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
            FloatOperationIr::Powf(desc) => FloatOperationIr::Powf(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
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
            FloatOperationIr::Tan(desc) => FloatOperationIr::Tan(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Cosh(desc) => FloatOperationIr::Cosh(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Sinh(desc) => FloatOperationIr::Sinh(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::ArcCos(desc) => FloatOperationIr::ArcCos(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::ArcCosh(desc) => FloatOperationIr::ArcCosh(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::ArcSin(desc) => FloatOperationIr::ArcSin(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::ArcSinh(desc) => FloatOperationIr::ArcSinh(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::ArcTan(desc) => FloatOperationIr::ArcTan(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::ArcTanh(desc) => FloatOperationIr::ArcTanh(UnaryOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::ArcTan2(desc) => FloatOperationIr::ArcTan2(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::IntoInt(desc) => FloatOperationIr::IntoInt(CastOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationIr::Matmul(desc) => FloatOperationIr::Matmul(MatmulOpIr {
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
            FloatOperationIr::Trunc(desc) => FloatOperationIr::Ceil(UnaryOpIr {
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
            FloatOperationIr::GridSample2d(desc) => {
                FloatOperationIr::GridSample2d(GridSample2dOpIr {
                    tensor: desc.tensor.to_relative(converter),
                    grid: desc.grid.to_relative(converter),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
        }
    }
}

impl RelativeOps for BoolOperationIr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            BoolOperationIr::IntoFloat(desc) => BoolOperationIr::IntoFloat(CastOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BoolOperationIr::IntoInt(desc) => BoolOperationIr::IntoInt(CastOpIr {
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
            IntOperationIr::IntoFloat(desc) => IntOperationIr::IntoFloat(CastOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            IntOperationIr::Matmul(desc) => IntOperationIr::Matmul(MatmulOpIr {
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
            NumericOperationIr::Full(desc) => NumericOperationIr::Full(FullOpIr {
                out: desc.out.to_relative(converter),
                value: desc.value.to_relative(converter),
            }),
            NumericOperationIr::MeanDim(desc) => NumericOperationIr::MeanDim(ReduceDimOpIr {
                input: desc.input.to_relative(converter),
                axis: desc.axis,
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::Mean(desc) => NumericOperationIr::Mean(ReduceOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::Sum(desc) => NumericOperationIr::Sum(ReduceOpIr {
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
            NumericOperationIr::Prod(desc) => NumericOperationIr::Prod(ReduceOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::ProdDim(desc) => NumericOperationIr::ProdDim(ReduceDimOpIr {
                input: desc.input.to_relative(converter),
                axis: desc.axis,
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
            NumericOperationIr::Max(desc) => NumericOperationIr::Max(ReduceOpIr {
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
            NumericOperationIr::Min(desc) => NumericOperationIr::Min(ReduceOpIr {
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
            NumericOperationIr::MaxAbs(desc) => NumericOperationIr::MaxAbs(ReduceOpIr {
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
            NumericOperationIr::Powi(desc) => NumericOperationIr::Powi(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::PowiScalar(desc) => NumericOperationIr::PowiScalar(ScalarOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationIr::CumSum(desc) => NumericOperationIr::CumSum(DimOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
                axis: desc.axis,
            }),
            NumericOperationIr::CumProd(desc) => NumericOperationIr::CumProd(DimOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
                axis: desc.axis,
            }),
            NumericOperationIr::CumMin(desc) => NumericOperationIr::CumMin(DimOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
                axis: desc.axis,
            }),
            NumericOperationIr::CumMax(desc) => NumericOperationIr::CumMax(DimOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
                axis: desc.axis,
            }),
        }
    }
}

impl RelativeOps for BaseOperationIr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            BaseOperationIr::Reshape(desc) => BaseOperationIr::Reshape(ShapeOpIr {
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
            BaseOperationIr::Expand(desc) => BaseOperationIr::Expand(ShapeOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
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
                ranges: desc.ranges.iter().map(|_info| Slice::from(0..1)).collect(),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationIr::SliceAssign(desc) => BaseOperationIr::SliceAssign(SliceAssignOpIr {
                tensor: desc.tensor.to_relative(converter),
                ranges: desc.ranges.iter().map(|_range| Slice::from(0..1)).collect(),
                value: desc.value.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationIr::Gather(desc) => BaseOperationIr::Gather(GatherOpIr {
                tensor: desc.tensor.to_relative(converter),
                dim: desc.dim,
                indices: desc.indices.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationIr::Scatter(desc) => BaseOperationIr::Scatter(ScatterOpIr {
                tensor: desc.tensor.to_relative(converter),
                dim: desc.dim,
                indices: desc.indices.to_relative(converter),
                value: desc.value.to_relative(converter),
                update: desc.update,
                out: desc.out.to_relative(converter),
            }),
            BaseOperationIr::Select(desc) => BaseOperationIr::Select(SelectOpIr {
                tensor: desc.tensor.to_relative(converter),
                dim: desc.dim,
                indices: desc.indices.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationIr::SelectAssign(desc) => {
                BaseOperationIr::SelectAssign(SelectAssignOpIr {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    indices: desc.indices.to_relative(converter),
                    value: desc.value.to_relative(converter),
                    update: desc.update,
                    out: desc.out.to_relative(converter),
                })
            }
            BaseOperationIr::MaskWhere(desc) => BaseOperationIr::MaskWhere(MaskWhereOpIr {
                tensor: desc.tensor.to_relative(converter),
                mask: desc.mask.to_relative(converter),
                value: desc.value.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationIr::MaskFill(desc) => BaseOperationIr::MaskFill(MaskFillOpIr {
                tensor: desc.tensor.to_relative(converter),
                mask: desc.mask.to_relative(converter),
                value: desc.value.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationIr::Equal(desc) => BaseOperationIr::Equal(BinaryOpIr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationIr::EqualElem(desc) => BaseOperationIr::EqualElem(ScalarOpIr {
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
            BaseOperationIr::Cast(desc) => BaseOperationIr::Cast(CastOpIr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationIr::Empty(desc) => BaseOperationIr::Empty(desc.to_relative(converter)),
            BaseOperationIr::Ones(desc) => BaseOperationIr::Ones(desc.to_relative(converter)),
            BaseOperationIr::Zeros(desc) => BaseOperationIr::Zeros(desc.to_relative(converter)),
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

impl RelativeOps for CreationOpIr {
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
        let mut relative_shape = Vec::with_capacity(self.shape.rank());
        for dim in self.shape.iter() {
            if let Some(dim_id) = converter.shapes_global2relative.get(dim) {
                // We already saw that dim value before, so we retrieve its ID.
                relative_shape.push(*dim_id);
            } else {
                // We never saw this dim value before, therefore we create a new ID.
                let dim_id = converter.shapes_global2relative.len();
                relative_shape.push(dim_id);

                converter.shapes_global2relative.insert(*dim, dim_id);
                converter.shapes_relative2global.insert(dim_id, *dim);
            }
        }

        // We create the relative tensor.
        let relative_tensor = TensorIr {
            id: relative_id,
            shape: Shape::from(relative_shape),
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

#[cfg(feature = "distributed")]
impl RelativeOps for DistributedOperationIr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            DistributedOperationIr::AllReduce(desc) => {
                DistributedOperationIr::AllReduce(AllReduceOpIr {
                    tensor: desc.tensor.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
        }
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
        ScalarIr::UInt(id.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::DType;
    use burn_ir::{TensorId, TensorIr, TensorStatus};

    /// Helper to build a minimal [`Context`] with string handles for testing fork behavior.
    fn make_test_context() -> Context<String> {
        let mut ctx = Context {
            tensors: HashMap::new(),
            handles: HandleContainer::new(),
            scalars: HashMap::new(),
            shapes_relative2global: HashMap::new(),
        };

        let id_input = TensorId::new(1);
        ctx.tensors.insert(
            id_input,
            TensorIr {
                id: id_input,
                shape: Shape::new([4, 4]),
                status: TensorStatus::ReadOnly,
                dtype: DType::F32,
            },
        );
        ctx.handles
            .register_handle(id_input, "input_handle".to_string());

        ctx
    }

    #[test]
    fn context_fork_output_handles_are_isolated() {
        // Output handles registered in a forked context are NOT visible in the original —
        // forks are independent sandboxes. `burn-cubecl-fusion`'s `TuneInput` layer is
        // responsible for merging new handles back into the real context when appropriate.
        let original = make_test_context();
        let output_id = TensorId::new(100);

        {
            let mut fork = original.fork();
            fork.handles
                .register_handle(output_id, "output_handle".to_string());

            // The fork has the output handle.
            assert!(fork.handles.get_handle_ref(&output_id).is_some());
        }

        // But the original does NOT — isolation is the point of `fork()`.
        assert!(original.handles.get_handle_ref(&output_id).is_none());
    }

    #[test]
    fn context_fork_preserves_input_handles() {
        let original = make_test_context();
        let input_id = TensorId::new(1);

        let fork = original.fork();

        // Fork should have a copy of the input handle.
        assert_eq!(
            fork.handles.get_handle_ref(&input_id),
            Some(&"input_handle".to_string())
        );
        // Original is unchanged.
        assert_eq!(
            original.handles.get_handle_ref(&input_id),
            Some(&"input_handle".to_string())
        );
    }

    #[test]
    fn context_double_fork_fully_isolated() {
        // Forking a fork creates a second level of isolation — a mutation in `fork2` is
        // invisible to both `fork1` and the original.
        let original = make_test_context();

        let fork1 = original.fork();
        let mut fork2 = fork1.fork();

        let deep_output_id = TensorId::new(200);
        fork2
            .handles
            .register_handle(deep_output_id, "deep_output".to_string());

        // Neither the first fork nor the original see the deeply-nested output.
        assert!(fork1.handles.get_handle_ref(&deep_output_id).is_none());
        assert!(original.handles.get_handle_ref(&deep_output_id).is_none());
    }
}

#[cfg(test)]
mod tests_ir {
    use super::*;
    use burn_backend::DType;
    use burn_ir::{TensorId, TensorIr, TensorStatus};

    #[test]
    fn tensor_description_to_relative() {
        let tensor1 = TensorIr {
            id: TensorId::new(500),
            shape: Shape::new([512, 32, 2048]),
            status: TensorStatus::ReadOnly,
            dtype: DType::F32,
        };
        let tensor2 = TensorIr {
            id: TensorId::new(501),
            shape: Shape::new([512, 128, 2048]),
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
                shape: Shape::new([1, 2, 3]),
                status: TensorStatus::ReadOnly,
                dtype: DType::F32
            }
        );
        assert_eq!(
            tensor2_local,
            TensorIr {
                id: TensorId::new(1),
                shape: Shape::new([1, 4, 3]),
                status: TensorStatus::ReadOnly,
                dtype: DType::F32
            }
        );
    }

    #[test]
    fn scalar_ir_to_relative() {
        let scalar1 = ScalarIr::Float(1.0);
        let scalar2 = ScalarIr::UInt(1);
        let mut converter = OperationConverter::default();
        let scalar1_local = scalar1.to_relative(&mut converter);
        let scalar2_local = scalar2.to_relative(&mut converter);

        assert_eq!(scalar1_local, ScalarIr::UInt(0));
        assert_eq!(scalar2_local, ScalarIr::UInt(1));
    }
}
