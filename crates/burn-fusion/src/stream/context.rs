use burn_tensor::{repr::*, DType, Element, ElementConversion};
use half::{bf16, f16};
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
    /// The tensor mapping where local tensor id points to the updated tensor description.
    pub tensors: &'a mut HashMap<TensorId, TensorDescription>,
    /// Handle container to retrieve tensors based on their description.
    pub handles: &'a mut HandleContainer<H>,
    /// F32 scalars found in the graph in the order they appeared.
    pub scalar_f32: &'a Vec<f32>,
    /// F16 scalars found in the graph in the order they appeared.
    pub scalar_f16: &'a Vec<f16>,
    /// BF16 scalars found in the graph in the order they appeared.
    pub scalar_bf16: &'a Vec<bf16>,
    /// i64 scalars found in the graph in the order they appeared.
    pub scalar_i64: &'a Vec<i64>,
    /// i32 scalars found in the graph in the order they appeared.
    pub scalar_i32: &'a Vec<i32>,
    /// i16 scalars found in the graph in the order they appeared.
    pub scalar_i16: &'a Vec<i16>,
    /// i8 scalars found in the graph in the order they appeared.
    pub scalar_i8: &'a Vec<i8>,
    /// u64 scalars found in the graph in the order they appeared.
    pub scalar_u64: &'a Vec<u64>,
    /// u32 scalars found in the graph in the order they appeared.
    pub scalar_u32: &'a Vec<u32>,
    /// u16 scalars found in the graph in the order they appeared.
    pub scalar_u16: &'a Vec<u16>,
    /// u8 scalars found in the graph in the order they appeared.
    pub scalar_u8: &'a Vec<u8>,
}

#[derive(Default)]
pub(crate) struct OperationConverter {
    tensors_relative2global: HashMap<TensorId, TensorDescription>,
    tensors_global2relative: HashMap<TensorId, TensorDescription>,
    /// Only useful to create new shape ID.
    /// You should use tensor descriptions to retrieve the proper shape.
    shapes_global2relative: HashMap<usize, usize>,
    scalar_f32: Vec<f32>,
    scalar_f16: Vec<f16>,
    scalar_bf16: Vec<bf16>,
    scalar_i64: Vec<i64>,
    scalar_i32: Vec<i32>,
    scalar_i16: Vec<i16>,
    scalar_i8: Vec<i8>,
    scalar_u64: Vec<u64>,
    scalar_u32: Vec<u32>,
    scalar_u16: Vec<u16>,
    scalar_u8: Vec<u8>,
}

/// Fork of a [context](Context) which owns its data.
pub struct ContextOwned<H> {
    tensors: HashMap<TensorId, TensorDescription>,
    handles: HandleContainer<H>,
    scalar_f32: Vec<f32>,
    scalar_f16: Vec<f16>,
    scalar_bf16: Vec<bf16>,
    scalar_i64: Vec<i64>,
    scalar_i32: Vec<i32>,
    scalar_i16: Vec<i16>,
    scalar_i8: Vec<i8>,
    scalar_u64: Vec<u64>,
    scalar_u32: Vec<u32>,
    scalar_u16: Vec<u16>,
    scalar_u8: Vec<u8>,
}

impl<H: Clone> ContextOwned<H> {
    /// Convert into [context](Context).
    pub fn as_context(&mut self) -> Context<'_, H> {
        Context {
            tensors: &mut self.tensors,
            handles: &mut self.handles,
            scalar_f32: &self.scalar_f32,
            scalar_f16: &self.scalar_f16,
            scalar_bf16: &self.scalar_bf16,
            scalar_i64: &self.scalar_i64,
            scalar_i32: &self.scalar_i32,
            scalar_i16: &self.scalar_i16,
            scalar_i8: &self.scalar_i8,
            scalar_u64: &self.scalar_u64,
            scalar_u32: &self.scalar_u32,
            scalar_u16: &self.scalar_u16,
            scalar_u8: &self.scalar_u8,
        }
    }

    /// Fork the context again.
    pub fn fork(&self) -> ContextOwned<H> {
        ContextOwned {
            tensors: self.tensors.clone(),
            handles: self.handles.fork(),
            scalar_f32: self.scalar_f32.clone(),
            scalar_f16: self.scalar_f16.clone(),
            scalar_bf16: self.scalar_bf16.clone(),
            scalar_i64: self.scalar_i64.clone(),
            scalar_i32: self.scalar_i32.clone(),
            scalar_i16: self.scalar_i16.clone(),
            scalar_i8: self.scalar_i8.clone(),
            scalar_u64: self.scalar_u64.clone(),
            scalar_u32: self.scalar_u32.clone(),
            scalar_u16: self.scalar_u16.clone(),
            scalar_u8: self.scalar_u8.clone(),
        }
    }
}

impl<H: Clone> Context<'_, H> {
    /// Fork the context into an [owned context](ContextOwned).
    pub fn fork(&self) -> ContextOwned<H> {
        ContextOwned {
            tensors: self.tensors.clone(),
            handles: self.handles.fork(),
            scalar_f32: self.scalar_f32.clone(),
            scalar_f16: self.scalar_f16.clone(),
            scalar_bf16: self.scalar_bf16.clone(),
            scalar_i64: self.scalar_i64.clone(),
            scalar_i32: self.scalar_i32.clone(),
            scalar_i16: self.scalar_i16.clone(),
            scalar_i8: self.scalar_i8.clone(),
            scalar_u64: self.scalar_u64.clone(),
            scalar_u32: self.scalar_u32.clone(),
            scalar_u16: self.scalar_u16.clone(),
            scalar_u8: self.scalar_u8.clone(),
        }
    }
}

pub(crate) trait RelativeOps {
    /// Convert (usually an [`OperationDescription`]) to a relative form.
    ///
    /// The id and the shape of tensors will be computed relative to existing
    /// operations in the queue. We do this because we want to fuse operations
    /// that have similar shapes, but we do not care about the exact values.
    ///
    /// Similar we do not care about the exact ids of the tensor, but about their
    /// relative ids (how close they are in the operation queue)
    fn to_relative(&self, converter: &mut OperationConverter) -> Self;
}

trait RelativeOpsScalar<E: Element> {
    fn to_relative<F>(&self, converter: &mut OperationConverter, local_elem: F) -> Self
    where
        F: Fn(&mut OperationConverter, &E) -> E;
}

impl OperationConverter {
    pub(crate) fn context<'a, H>(
        &'a mut self,
        handles: &'a mut HandleContainer<H>,
    ) -> Context<'a, H> {
        Context {
            handles,
            tensors: &mut self.tensors_relative2global,
            scalar_f32: &self.scalar_f32,
            scalar_f16: &self.scalar_f16,
            scalar_bf16: &self.scalar_bf16,
            scalar_i64: &self.scalar_i64,
            scalar_i32: &self.scalar_i32,
            scalar_i16: &self.scalar_i16,
            scalar_i8: &self.scalar_i8,
            scalar_u64: &self.scalar_u64,
            scalar_u32: &self.scalar_u32,
            scalar_u16: &self.scalar_u16,
            scalar_u8: &self.scalar_u8,
        }
    }

    pub(crate) fn clear(&mut self) {
        self.tensors_relative2global.clear();
        self.tensors_global2relative.clear();
        self.shapes_global2relative.clear();
        self.scalar_f32.clear();
        self.scalar_f16.clear();
        self.scalar_bf16.clear();
        self.scalar_i64.clear();
        self.scalar_i32.clear();
        self.scalar_i16.clear();
        self.scalar_i8.clear();
        self.scalar_u64.clear();
        self.scalar_u32.clear();
        self.scalar_u16.clear();
        self.scalar_u8.clear();
    }

    pub(crate) fn relative_float<E: Element>(&mut self, elem: &E, dtype: &DType) -> E {
        match dtype {
            burn_tensor::DType::F32 => self.scalar_f32.push(elem.elem()),
            burn_tensor::DType::F16 => self.scalar_f16.push(elem.elem()),
            burn_tensor::DType::BF16 => self.scalar_bf16.push(elem.elem()),
            _ => todo!("Unsupported"),
        }

        // We return 0 so that the id from a scalar operation is the same no matter its scalar
        // value.
        0.elem()
    }

    pub(crate) fn relative_int<E: Element>(&mut self, elem: &E, dtype: &DType) -> E {
        match dtype {
            DType::I64 => self.scalar_i64.push(elem.elem()),
            DType::I32 => self.scalar_i32.push(elem.elem()),
            DType::I16 => self.scalar_i16.push(elem.elem()),
            DType::I8 => self.scalar_i8.push(elem.elem()),
            DType::U64 => self.scalar_u64.push(elem.elem()),
            DType::U32 => self.scalar_u32.push(elem.elem()),
            DType::U16 => self.scalar_u16.push(elem.elem()),
            DType::U8 => self.scalar_u8.push(elem.elem()),
            _ => todo!("Unsupported"),
        }
        // We return 0 so that the id from a scalar operation is the same no matter its scalar
        // value.
        0.elem()
    }
}

impl RelativeOps for OperationDescription {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            OperationDescription::BaseFloat(ops) => {
                OperationDescription::BaseFloat(ops.to_relative(converter))
            }
            OperationDescription::BaseInt(ops) => {
                OperationDescription::BaseInt(ops.to_relative(converter))
            }
            OperationDescription::BaseBool(ops) => {
                OperationDescription::BaseBool(ops.to_relative(converter))
            }
            OperationDescription::NumericFloat(dtype, ops) => OperationDescription::NumericFloat(
                *dtype,
                ops.to_relative(converter, |converter, e| converter.relative_float(e, dtype)),
            ),
            OperationDescription::NumericInt(dtype, ops) => OperationDescription::NumericInt(
                *dtype,
                ops.to_relative(converter, |converter, e| converter.relative_int(e, dtype)),
            ),
            OperationDescription::Bool(ops) => {
                OperationDescription::Bool(ops.to_relative(converter))
            }
            OperationDescription::Int(ops) => OperationDescription::Int(ops.to_relative(converter)),
            OperationDescription::Float(dtype, ops) => OperationDescription::Float(
                *dtype,
                RelativeOpsScalar::<f32>::to_relative(ops, converter, |converter, e| {
                    converter.relative_float(e, dtype)
                }),
            ),
            OperationDescription::Module(ops) => {
                OperationDescription::Module(ops.to_relative(converter))
            }
            OperationDescription::Custom(ops) => {
                OperationDescription::Custom(ops.to_relative(converter))
            }
        }
    }
}

impl RelativeOps for ModuleOperationDescription {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            ModuleOperationDescription::Embedding(desc) => {
                ModuleOperationDescription::Embedding(EmbeddingDescription {
                    weights: desc.weights.to_relative(converter),
                    indices: desc.indices.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::EmbeddingBackward(desc) => {
                ModuleOperationDescription::EmbeddingBackward(EmbeddingBackwardDescription {
                    weights: desc.weights.to_relative(converter),
                    out_grad: desc.out_grad.to_relative(converter),
                    indices: desc.indices.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::Conv1d(desc) => {
                ModuleOperationDescription::Conv1d(Conv1dDescription {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::Conv2d(desc) => {
                ModuleOperationDescription::Conv2d(Conv2dDescription {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::Conv3d(desc) => {
                ModuleOperationDescription::Conv3d(Conv3dDescription {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::DeformableConv2d(desc) => {
                ModuleOperationDescription::DeformableConv2d(Box::new(DeformConv2dDescription {
                    x: desc.x.to_relative(converter),
                    offset: desc.offset.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    mask: desc.mask.as_ref().map(|t| t.to_relative(converter)),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                }))
            }
            ModuleOperationDescription::DeformableConv2dBackward(desc) => {
                ModuleOperationDescription::DeformableConv2dBackward(Box::new(
                    DeformConv2dBackwardDescription {
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
                    },
                ))
            }
            ModuleOperationDescription::ConvTranspose1d(desc) => {
                ModuleOperationDescription::ConvTranspose1d(ConvTranspose1dDescription {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::ConvTranspose2d(desc) => {
                ModuleOperationDescription::ConvTranspose2d(ConvTranspose2dDescription {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::ConvTranspose3d(desc) => {
                ModuleOperationDescription::ConvTranspose3d(ConvTranspose3dDescription {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::AvgPool1d(desc) => {
                ModuleOperationDescription::AvgPool1d(AvgPool1dDescription {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::AvgPool2d(desc) => {
                ModuleOperationDescription::AvgPool2d(AvgPool2dDescription {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::AvgPool1dBackward(desc) => {
                ModuleOperationDescription::AvgPool1dBackward(AvgPool1dBackwardDescription {
                    x: desc.x.to_relative(converter),
                    grad: desc.grad.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::AvgPool2dBackward(desc) => {
                ModuleOperationDescription::AvgPool2dBackward(AvgPool2dBackwardDescription {
                    x: desc.x.to_relative(converter),
                    grad: desc.grad.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::AdaptiveAvgPool1d(desc) => {
                ModuleOperationDescription::AdaptiveAvgPool1d(AdaptiveAvgPool1dDescription {
                    x: desc.x.to_relative(converter),
                    output_size: desc.output_size,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::AdaptiveAvgPool2d(desc) => {
                ModuleOperationDescription::AdaptiveAvgPool2d(AdaptiveAvgPool2dDescription {
                    x: desc.x.to_relative(converter),
                    output_size: desc.output_size,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::AdaptiveAvgPool1dBackward(desc) => {
                ModuleOperationDescription::AdaptiveAvgPool1dBackward(
                    AdaptiveAvgPool1dBackwardDescription {
                        x: desc.x.to_relative(converter),
                        grad: desc.grad.to_relative(converter),
                        out: desc.out.to_relative(converter),
                    },
                )
            }
            ModuleOperationDescription::AdaptiveAvgPool2dBackward(desc) => {
                ModuleOperationDescription::AdaptiveAvgPool2dBackward(
                    AdaptiveAvgPool2dBackwardDescription {
                        x: desc.x.to_relative(converter),
                        grad: desc.grad.to_relative(converter),
                        out: desc.out.to_relative(converter),
                    },
                )
            }
            ModuleOperationDescription::MaxPool1d(desc) => {
                ModuleOperationDescription::MaxPool1d(MaxPool1dDescription {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::MaxPool1dWithIndices(desc) => {
                ModuleOperationDescription::MaxPool1dWithIndices(MaxPool1dWithIndicesDescription {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_relative(converter),
                    out_indices: desc.out_indices.to_relative(converter),
                })
            }
            ModuleOperationDescription::MaxPool1dWithIndicesBackward(desc) => {
                ModuleOperationDescription::MaxPool1dWithIndicesBackward(
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
            ModuleOperationDescription::MaxPool2d(desc) => {
                ModuleOperationDescription::MaxPool2d(MaxPool2dDescription {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::MaxPool2dWithIndices(desc) => {
                ModuleOperationDescription::MaxPool2dWithIndices(MaxPool2dWithIndicesDescription {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_relative(converter),
                    out_indices: desc.out_indices.to_relative(converter),
                })
            }
            ModuleOperationDescription::MaxPool2dWithIndicesBackward(desc) => {
                ModuleOperationDescription::MaxPool2dWithIndicesBackward(
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
            ModuleOperationDescription::Interpolate(desc) => {
                ModuleOperationDescription::Interpolate(InterpolateDescription {
                    x: desc.x.to_relative(converter),
                    output_size: desc.output_size,
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationDescription::InterpolateBackward(desc) => {
                ModuleOperationDescription::InterpolateBackward(InterpolateBackwardDescription {
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

impl RelativeOpsScalar<f32> for FloatOperationDescription {
    fn to_relative<F>(&self, converter: &mut OperationConverter, local_elem: F) -> Self
    where
        F: Fn(&mut OperationConverter, &f32) -> f32,
    {
        match self {
            FloatOperationDescription::Exp(desc) => {
                FloatOperationDescription::Exp(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationDescription::Log(desc) => {
                FloatOperationDescription::Log(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationDescription::Log1p(desc) => {
                FloatOperationDescription::Log1p(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationDescription::Erf(desc) => {
                FloatOperationDescription::Erf(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationDescription::PowfScalar(desc) => {
                FloatOperationDescription::PowfScalar(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs.elem()),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationDescription::Sqrt(desc) => {
                FloatOperationDescription::Sqrt(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationDescription::Cos(desc) => {
                FloatOperationDescription::Cos(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationDescription::Sin(desc) => {
                FloatOperationDescription::Sin(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationDescription::Tanh(desc) => {
                FloatOperationDescription::Tanh(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationDescription::IntoInt(desc) => {
                FloatOperationDescription::IntoInt(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationDescription::Matmul(desc) => {
                FloatOperationDescription::Matmul(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationDescription::Random(desc) => {
                FloatOperationDescription::Random(RandomOperationDescription {
                    out: desc.out.to_relative(converter),
                    distribution: desc.distribution,
                })
            }
            FloatOperationDescription::Recip(desc) => {
                FloatOperationDescription::Recip(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationDescription::Quantize(desc) => {
                FloatOperationDescription::Quantize(QuantizeOperationDescription {
                    tensor: desc.tensor.to_relative(converter),
                    qparams: QuantizationParametersDescription {
                        scale: desc.qparams.scale.to_relative(converter),
                        offset: desc
                            .qparams
                            .offset
                            .as_ref()
                            .map(|x| x.to_relative(converter)),
                    },
                    scheme: desc.scheme,
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationDescription::Dequantize(desc) => {
                FloatOperationDescription::Dequantize(DequantizeOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationDescription::Round(desc) => {
                FloatOperationDescription::Round(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationDescription::Floor(desc) => {
                FloatOperationDescription::Floor(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationDescription::Ceil(desc) => {
                FloatOperationDescription::Ceil(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
        }
    }
}

impl RelativeOps for BoolOperationDescription {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            BoolOperationDescription::IntoFloat(desc) => {
                BoolOperationDescription::IntoFloat(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            BoolOperationDescription::IntoInt(desc) => {
                BoolOperationDescription::IntoInt(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            BoolOperationDescription::Not(desc) => {
                BoolOperationDescription::Not(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
        }
    }
}

impl RelativeOps for IntOperationDescription {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            IntOperationDescription::IntoFloat(desc) => {
                IntOperationDescription::IntoFloat(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationDescription::BitwiseAnd(desc) => {
                IntOperationDescription::BitwiseAnd(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationDescription::BitwiseAndScalar(desc) => {
                IntOperationDescription::BitwiseAndScalar(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationDescription::BitwiseOr(desc) => {
                IntOperationDescription::BitwiseOr(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationDescription::BitwiseOrScalar(desc) => {
                IntOperationDescription::BitwiseOrScalar(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationDescription::BitwiseXor(desc) => {
                IntOperationDescription::BitwiseXor(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationDescription::BitwiseXorScalar(desc) => {
                IntOperationDescription::BitwiseXorScalar(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationDescription::BitwiseNot(desc) => {
                IntOperationDescription::BitwiseNot(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationDescription::BitwiseLeftShift(desc) => {
                IntOperationDescription::BitwiseLeftShift(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationDescription::BitwiseLeftShiftScalar(desc) => {
                IntOperationDescription::BitwiseLeftShiftScalar(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationDescription::BitwiseRightShift(desc) => {
                IntOperationDescription::BitwiseRightShift(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationDescription::BitwiseRightShiftScalar(desc) => {
                IntOperationDescription::BitwiseRightShiftScalar(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
        }
    }
}

impl RelativeOps for CustomOpDescription {
    fn to_relative(&self, converter: &mut OperationConverter) -> CustomOpDescription {
        let id = self.id.clone();

        CustomOpDescription {
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

impl<E: Element> RelativeOpsScalar<E> for NumericOperationDescription<E> {
    fn to_relative<F>(&self, converter: &mut OperationConverter, local_elem: F) -> Self
    where
        F: Fn(&mut OperationConverter, &E) -> E,
    {
        match self {
            NumericOperationDescription::Add(desc) => {
                NumericOperationDescription::Add(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::AddScalar(desc) => {
                NumericOperationDescription::AddScalar(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::Sub(desc) => {
                NumericOperationDescription::Sub(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::SubScalar(desc) => {
                NumericOperationDescription::SubScalar(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::Div(desc) => {
                NumericOperationDescription::Div(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::DivScalar(desc) => {
                NumericOperationDescription::DivScalar(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::Rem(desc) => {
                NumericOperationDescription::Rem(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::RemScalar(desc) => {
                NumericOperationDescription::RemScalar(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::Mul(desc) => {
                NumericOperationDescription::Mul(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::MulScalar(desc) => {
                NumericOperationDescription::MulScalar(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::Abs(desc) => {
                NumericOperationDescription::Abs(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::Ones(desc) => {
                NumericOperationDescription::Ones(desc.to_relative(converter))
            }
            NumericOperationDescription::Zeros(desc) => {
                NumericOperationDescription::Zeros(desc.to_relative(converter))
            }
            NumericOperationDescription::Full(desc) => NumericOperationDescription::Full((
                desc.0.to_relative(converter),
                local_elem(converter, &desc.1),
            )),
            NumericOperationDescription::Gather(desc) => {
                NumericOperationDescription::Gather(GatherOperationDescription {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    indices: desc.indices.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::Scatter(desc) => {
                NumericOperationDescription::Scatter(ScatterOperationDescription {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    indices: desc.indices.to_relative(converter),
                    value: desc.value.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::Select(desc) => {
                NumericOperationDescription::Select(SelectOperationDescription {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    indices: desc.indices.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::SelectAssign(desc) => {
                NumericOperationDescription::SelectAssign(SelectAssignOperationDescription {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    indices: desc.indices.to_relative(converter),
                    value: desc.value.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::MaskWhere(desc) => {
                NumericOperationDescription::MaskWhere(MaskWhereOperationDescription {
                    tensor: desc.tensor.to_relative(converter),
                    mask: desc.mask.to_relative(converter),
                    value: desc.value.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::MaskFill(desc) => {
                NumericOperationDescription::MaskFill(MaskFillOperationDescription {
                    tensor: desc.tensor.to_relative(converter),
                    mask: desc.mask.to_relative(converter),
                    value: local_elem(converter, &desc.value),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::MeanDim(desc) => {
                NumericOperationDescription::MeanDim(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs, // Dim should stay the same.
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::Mean(desc) => {
                NumericOperationDescription::Mean(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::Sum(desc) => {
                NumericOperationDescription::Sum(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::SumDim(desc) => {
                NumericOperationDescription::SumDim(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs, // Dim should stay the same.
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::Prod(desc) => {
                NumericOperationDescription::Prod(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::ProdDim(desc) => {
                NumericOperationDescription::ProdDim(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs, // Dim should stay the same.
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::EqualElem(desc) => {
                NumericOperationDescription::EqualElem(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::Greater(desc) => {
                NumericOperationDescription::Greater(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::GreaterElem(desc) => {
                NumericOperationDescription::GreaterElem(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::GreaterEqual(desc) => {
                NumericOperationDescription::GreaterEqual(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::GreaterEqualElem(desc) => {
                NumericOperationDescription::GreaterEqualElem(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::Lower(desc) => {
                NumericOperationDescription::Lower(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::LowerElem(desc) => {
                NumericOperationDescription::LowerElem(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::LowerEqual(desc) => {
                NumericOperationDescription::LowerEqual(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::LowerEqualElem(desc) => {
                NumericOperationDescription::LowerEqualElem(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::ArgMax(desc) => {
                NumericOperationDescription::ArgMax(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::ArgMin(desc) => {
                NumericOperationDescription::ArgMin(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::Max(desc) => {
                NumericOperationDescription::Max(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::MaxDimWithIndices(desc) => {
                NumericOperationDescription::MaxDimWithIndices(ReduceDimWithIndicesDescription {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    out: desc.out.to_relative(converter),
                    out_indices: desc.out_indices.to_relative(converter),
                })
            }
            NumericOperationDescription::MinDimWithIndices(desc) => {
                NumericOperationDescription::MinDimWithIndices(ReduceDimWithIndicesDescription {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    out: desc.out.to_relative(converter),
                    out_indices: desc.out_indices.to_relative(converter),
                })
            }
            NumericOperationDescription::Min(desc) => {
                NumericOperationDescription::Min(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::MaxDim(desc) => {
                NumericOperationDescription::MaxDim(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::MinDim(desc) => {
                NumericOperationDescription::MinDim(ScalarOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::Clamp(desc) => {
                NumericOperationDescription::Clamp(ClampOperationDescription {
                    tensor: desc.tensor.to_relative(converter),
                    min: local_elem(converter, &desc.min),
                    max: local_elem(converter, &desc.max),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationDescription::IntRandom(desc) => {
                NumericOperationDescription::IntRandom(RandomOperationDescription {
                    out: desc.out.to_relative(converter),
                    distribution: desc.distribution,
                })
            }
            NumericOperationDescription::Powf(desc) => {
                NumericOperationDescription::Powf(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
        }
    }
}

impl RelativeOps for BaseOperationDescription {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            BaseOperationDescription::ToDevice(desc) => {
                BaseOperationDescription::ToDevice(desc.to_relative(converter))
            }
            BaseOperationDescription::Reshape(desc) => {
                BaseOperationDescription::Reshape(ReshapeDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            BaseOperationDescription::SwapDims(desc) => {
                BaseOperationDescription::SwapDims(SwapDimsDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                    dim1: desc.dim1,
                    dim2: desc.dim2,
                })
            }
            BaseOperationDescription::Permute(desc) => {
                BaseOperationDescription::Permute(PermuteOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                    axes: desc.axes.clone(),
                })
            }
            BaseOperationDescription::Expand(desc) => {
                BaseOperationDescription::Expand(ExpandOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                    shape: desc.shape.clone(),
                })
            }
            BaseOperationDescription::Flip(desc) => {
                BaseOperationDescription::Flip(FlipOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                    axes: desc.axes.clone(),
                })
            }
            BaseOperationDescription::Slice(desc) => {
                BaseOperationDescription::Slice(SliceOperationDescription {
                    tensor: desc.tensor.to_relative(converter),
                    ranges: desc.ranges.iter().map(|_range| 0..1).collect(),
                    out: desc.out.to_relative(converter),
                })
            }
            BaseOperationDescription::SliceAssign(desc) => {
                BaseOperationDescription::SliceAssign(SliceAssignOperationDescription {
                    tensor: desc.tensor.to_relative(converter),
                    ranges: desc.ranges.iter().map(|_range| 0..1).collect(),
                    value: desc.value.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            BaseOperationDescription::Equal(desc) => {
                BaseOperationDescription::Equal(BinaryOperationDescription {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            BaseOperationDescription::RepeatDim(desc) => {
                BaseOperationDescription::RepeatDim(RepeatDimOperationDescription {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    times: desc.times,
                    out: desc.out.to_relative(converter),
                })
            }
            BaseOperationDescription::Cat(desc) => {
                BaseOperationDescription::Cat(CatOperationDescription {
                    tensors: desc
                        .tensors
                        .iter()
                        .map(|tensor| tensor.to_relative(converter))
                        .collect(),
                    dim: desc.dim,
                    out: desc.out.to_relative(converter),
                })
            }
            BaseOperationDescription::Cast(desc) => {
                BaseOperationDescription::Cast(UnaryOperationDescription {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            BaseOperationDescription::Empty(desc) => {
                BaseOperationDescription::Empty(desc.to_relative(converter))
            }
            BaseOperationDescription::FromData(desc) => {
                BaseOperationDescription::FromData(FromDataOperationDescription {
                    data: desc.data.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
        }
    }
}

impl RelativeOps for TensorDescription {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        let relative_id = if let Some(value) = converter.tensors_global2relative.get(&self.id) {
            // If we already have the same tensor registered, we have to update its value, but not
            // its id.
            value.id
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
            id: relative_id,
            shape: relative_shape,
            status: self.status.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_tensor::{
        repr::{TensorDescription, TensorId, TensorStatus},
        DType,
    };

    #[test]
    fn tensor_description_to_relative() {
        let tensor1 = TensorDescription {
            id: TensorId::new(500),
            shape: vec![512, 32, 2048],
            status: TensorStatus::ReadOnly,
            dtype: DType::F32,
        };
        let tensor2 = TensorDescription {
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
            TensorDescription {
                id: TensorId::new(0),
                shape: vec![0, 1, 2],
                status: TensorStatus::ReadOnly,
                dtype: DType::F32
            }
        );
        assert_eq!(
            tensor2_local,
            TensorDescription {
                id: TensorId::new(1),
                shape: vec![0, 3, 2],
                status: TensorStatus::ReadOnly,
                dtype: DType::F32
            }
        );
    }
}
