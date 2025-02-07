use burn_ir::*;
use burn_tensor::{DType, Element, ElementConversion};
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
    pub tensors: &'a mut HashMap<TensorId, TensorRepr>,
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

pub(crate) struct OperationConverter {
    tensors_relative2global: HashMap<TensorId, TensorRepr>,
    tensors_global2relative: HashMap<TensorId, TensorRepr>,
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

impl Default for OperationConverter {
    fn default() -> Self {
        let mut val = Self {
            tensors_relative2global: Default::default(),
            tensors_global2relative: Default::default(),
            shapes_global2relative: Default::default(),
            scalar_f32: Default::default(),
            scalar_f16: Default::default(),
            scalar_bf16: Default::default(),
            scalar_i64: Default::default(),
            scalar_i32: Default::default(),
            scalar_i16: Default::default(),
            scalar_i8: Default::default(),
            scalar_u64: Default::default(),
            scalar_u32: Default::default(),
            scalar_u16: Default::default(),
            scalar_u8: Default::default(),
        };

        // global 1 is always shape id 0.
        val.shapes_global2relative.insert(1, 0);

        val
    }
}

/// Fork of a [context](Context) which owns its data.
pub struct ContextOwned<H> {
    tensors: HashMap<TensorId, TensorRepr>,
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
    /// Convert (usually an [`OperationRepr`]) to a relative form.
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
        // global 1 is always shape id 0.
        self.shapes_global2relative.insert(1, 0);

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
            _ => todo!("Unsupported float dtype ({dtype:?}) for scalar ({elem:?})"),
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

impl RelativeOps for OperationRepr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            OperationRepr::BaseFloat(ops) => OperationRepr::BaseFloat(ops.to_relative(converter)),
            OperationRepr::BaseInt(ops) => OperationRepr::BaseInt(ops.to_relative(converter)),
            OperationRepr::BaseBool(ops) => OperationRepr::BaseBool(ops.to_relative(converter)),
            OperationRepr::NumericFloat(dtype, ops) => OperationRepr::NumericFloat(
                *dtype,
                ops.to_relative(converter, |converter, e| converter.relative_float(e, dtype)),
            ),
            OperationRepr::NumericInt(dtype, ops) => OperationRepr::NumericInt(
                *dtype,
                ops.to_relative(converter, |converter, e| converter.relative_int(e, dtype)),
            ),
            OperationRepr::Bool(ops) => OperationRepr::Bool(ops.to_relative(converter)),
            OperationRepr::Int(ops) => OperationRepr::Int(ops.to_relative(converter)),
            OperationRepr::Float(dtype, ops) => OperationRepr::Float(
                *dtype,
                RelativeOpsScalar::<f32>::to_relative(ops, converter, |converter, e| {
                    converter.relative_float(e, dtype)
                }),
            ),
            OperationRepr::Module(ops) => OperationRepr::Module(ops.to_relative(converter)),
            OperationRepr::Custom(ops) => OperationRepr::Custom(ops.to_relative(converter)),
            OperationRepr::Init(ops) => OperationRepr::Init(ops.to_relative(converter)),
        }
    }
}

impl RelativeOps for ModuleOperationRepr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            ModuleOperationRepr::Embedding(desc) => {
                ModuleOperationRepr::Embedding(EmbeddingOpRepr {
                    weights: desc.weights.to_relative(converter),
                    indices: desc.indices.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationRepr::EmbeddingBackward(desc) => {
                ModuleOperationRepr::EmbeddingBackward(EmbeddingBackwardOpRepr {
                    weights: desc.weights.to_relative(converter),
                    out_grad: desc.out_grad.to_relative(converter),
                    indices: desc.indices.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationRepr::Conv1d(desc) => ModuleOperationRepr::Conv1d(Conv1dOpRepr {
                x: desc.x.to_relative(converter),
                weight: desc.weight.to_relative(converter),
                bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                options: desc.options.clone(),
                out: desc.out.to_relative(converter),
            }),
            ModuleOperationRepr::Conv2d(desc) => ModuleOperationRepr::Conv2d(Conv2dOpRepr {
                x: desc.x.to_relative(converter),
                weight: desc.weight.to_relative(converter),
                bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                options: desc.options.clone(),
                out: desc.out.to_relative(converter),
            }),
            ModuleOperationRepr::Conv3d(desc) => ModuleOperationRepr::Conv3d(Conv3dOpRepr {
                x: desc.x.to_relative(converter),
                weight: desc.weight.to_relative(converter),
                bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                options: desc.options.clone(),
                out: desc.out.to_relative(converter),
            }),
            ModuleOperationRepr::DeformableConv2d(desc) => {
                ModuleOperationRepr::DeformableConv2d(Box::new(DeformConv2dOpRepr {
                    x: desc.x.to_relative(converter),
                    offset: desc.offset.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    mask: desc.mask.as_ref().map(|t| t.to_relative(converter)),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                }))
            }
            ModuleOperationRepr::DeformableConv2dBackward(desc) => {
                ModuleOperationRepr::DeformableConv2dBackward(Box::new(
                    DeformConv2dBackwardOpRepr {
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
            ModuleOperationRepr::ConvTranspose1d(desc) => {
                ModuleOperationRepr::ConvTranspose1d(ConvTranspose1dOpRepr {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationRepr::ConvTranspose2d(desc) => {
                ModuleOperationRepr::ConvTranspose2d(ConvTranspose2dOpRepr {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationRepr::ConvTranspose3d(desc) => {
                ModuleOperationRepr::ConvTranspose3d(ConvTranspose3dOpRepr {
                    x: desc.x.to_relative(converter),
                    weight: desc.weight.to_relative(converter),
                    bias: desc.bias.as_ref().map(|t| t.to_relative(converter)),
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationRepr::AvgPool1d(desc) => {
                ModuleOperationRepr::AvgPool1d(AvgPool1dOpRepr {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationRepr::AvgPool2d(desc) => {
                ModuleOperationRepr::AvgPool2d(AvgPool2dOpRepr {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationRepr::AvgPool1dBackward(desc) => {
                ModuleOperationRepr::AvgPool1dBackward(AvgPool1dBackwardOpRepr {
                    x: desc.x.to_relative(converter),
                    grad: desc.grad.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationRepr::AvgPool2dBackward(desc) => {
                ModuleOperationRepr::AvgPool2dBackward(AvgPool2dBackwardOpRepr {
                    x: desc.x.to_relative(converter),
                    grad: desc.grad.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    count_include_pad: desc.count_include_pad,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationRepr::AdaptiveAvgPool1d(desc) => {
                ModuleOperationRepr::AdaptiveAvgPool1d(AdaptiveAvgPool1dOpRepr {
                    x: desc.x.to_relative(converter),
                    output_size: desc.output_size,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationRepr::AdaptiveAvgPool2d(desc) => {
                ModuleOperationRepr::AdaptiveAvgPool2d(AdaptiveAvgPool2dOpRepr {
                    x: desc.x.to_relative(converter),
                    output_size: desc.output_size,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationRepr::AdaptiveAvgPool1dBackward(desc) => {
                ModuleOperationRepr::AdaptiveAvgPool1dBackward(AdaptiveAvgPool1dBackwardOpRepr {
                    x: desc.x.to_relative(converter),
                    grad: desc.grad.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationRepr::AdaptiveAvgPool2dBackward(desc) => {
                ModuleOperationRepr::AdaptiveAvgPool2dBackward(AdaptiveAvgPool2dBackwardOpRepr {
                    x: desc.x.to_relative(converter),
                    grad: desc.grad.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationRepr::MaxPool1d(desc) => {
                ModuleOperationRepr::MaxPool1d(MaxPool1dOpRepr {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationRepr::MaxPool1dWithIndices(desc) => {
                ModuleOperationRepr::MaxPool1dWithIndices(MaxPool1dWithIndicesOpRepr {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_relative(converter),
                    out_indices: desc.out_indices.to_relative(converter),
                })
            }
            ModuleOperationRepr::MaxPool1dWithIndicesBackward(desc) => {
                ModuleOperationRepr::MaxPool1dWithIndicesBackward(
                    MaxPool1dWithIndicesBackwardOpRepr {
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
            ModuleOperationRepr::MaxPool2d(desc) => {
                ModuleOperationRepr::MaxPool2d(MaxPool2dOpRepr {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationRepr::MaxPool2dWithIndices(desc) => {
                ModuleOperationRepr::MaxPool2dWithIndices(MaxPool2dWithIndicesOpRepr {
                    x: desc.x.to_relative(converter),
                    kernel_size: desc.kernel_size,
                    stride: desc.stride,
                    padding: desc.padding,
                    dilation: desc.dilation,
                    out: desc.out.to_relative(converter),
                    out_indices: desc.out_indices.to_relative(converter),
                })
            }
            ModuleOperationRepr::MaxPool2dWithIndicesBackward(desc) => {
                ModuleOperationRepr::MaxPool2dWithIndicesBackward(
                    MaxPool2dWithIndicesBackwardOpRepr {
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
            ModuleOperationRepr::Interpolate(desc) => {
                ModuleOperationRepr::Interpolate(InterpolateOpRepr {
                    x: desc.x.to_relative(converter),
                    output_size: desc.output_size,
                    options: desc.options.clone(),
                    out: desc.out.to_relative(converter),
                })
            }
            ModuleOperationRepr::InterpolateBackward(desc) => {
                ModuleOperationRepr::InterpolateBackward(InterpolateBackwardRepr {
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

impl RelativeOpsScalar<f32> for FloatOperationRepr {
    fn to_relative<F>(&self, converter: &mut OperationConverter, local_elem: F) -> Self
    where
        F: Fn(&mut OperationConverter, &f32) -> f32,
    {
        match self {
            FloatOperationRepr::Exp(desc) => FloatOperationRepr::Exp(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationRepr::Log(desc) => FloatOperationRepr::Log(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationRepr::Log1p(desc) => FloatOperationRepr::Log1p(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationRepr::Erf(desc) => FloatOperationRepr::Erf(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationRepr::PowfScalar(desc) => FloatOperationRepr::PowfScalar(ScalarOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: local_elem(converter, &desc.rhs.elem()),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationRepr::Sqrt(desc) => FloatOperationRepr::Sqrt(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationRepr::Cos(desc) => FloatOperationRepr::Cos(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationRepr::Sin(desc) => FloatOperationRepr::Sin(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationRepr::Tanh(desc) => FloatOperationRepr::Tanh(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationRepr::IntoInt(desc) => FloatOperationRepr::IntoInt(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationRepr::Matmul(desc) => FloatOperationRepr::Matmul(BinaryOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationRepr::Random(desc) => FloatOperationRepr::Random(RandomOpRepr {
                out: desc.out.to_relative(converter),
                distribution: desc.distribution,
            }),
            FloatOperationRepr::Recip(desc) => FloatOperationRepr::Recip(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationRepr::Quantize(desc) => FloatOperationRepr::Quantize(QuantizeOpRepr {
                tensor: desc.tensor.to_relative(converter),
                qparams: QuantizationParametersRepr {
                    scale: desc.qparams.scale.to_relative(converter),
                    offset: desc
                        .qparams
                        .offset
                        .as_ref()
                        .map(|x| x.to_relative(converter)),
                },
                scheme: desc.scheme,
                out: desc.out.to_relative(converter),
            }),
            FloatOperationRepr::Dequantize(desc) => {
                FloatOperationRepr::Dequantize(DequantizeOpRepr {
                    input: desc.input.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            FloatOperationRepr::Round(desc) => FloatOperationRepr::Round(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationRepr::Floor(desc) => FloatOperationRepr::Floor(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            FloatOperationRepr::Ceil(desc) => FloatOperationRepr::Ceil(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
        }
    }
}

impl RelativeOps for BoolOperationRepr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            BoolOperationRepr::IntoFloat(desc) => BoolOperationRepr::IntoFloat(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BoolOperationRepr::IntoInt(desc) => BoolOperationRepr::IntoInt(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BoolOperationRepr::Not(desc) => BoolOperationRepr::Not(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
        }
    }
}

impl RelativeOps for IntOperationRepr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            IntOperationRepr::IntoFloat(desc) => IntOperationRepr::IntoFloat(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            IntOperationRepr::BitwiseAnd(desc) => IntOperationRepr::BitwiseAnd(BinaryOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            IntOperationRepr::BitwiseAndScalar(desc) => {
                IntOperationRepr::BitwiseAndScalar(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationRepr::BitwiseOr(desc) => IntOperationRepr::BitwiseOr(BinaryOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            IntOperationRepr::BitwiseOrScalar(desc) => {
                IntOperationRepr::BitwiseOrScalar(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationRepr::BitwiseXor(desc) => IntOperationRepr::BitwiseXor(BinaryOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            IntOperationRepr::BitwiseXorScalar(desc) => {
                IntOperationRepr::BitwiseXorScalar(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationRepr::BitwiseNot(desc) => IntOperationRepr::BitwiseNot(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            IntOperationRepr::BitwiseLeftShift(desc) => {
                IntOperationRepr::BitwiseLeftShift(BinaryOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationRepr::BitwiseLeftShiftScalar(desc) => {
                IntOperationRepr::BitwiseLeftShiftScalar(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationRepr::BitwiseRightShift(desc) => {
                IntOperationRepr::BitwiseRightShift(BinaryOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            IntOperationRepr::BitwiseRightShiftScalar(desc) => {
                IntOperationRepr::BitwiseRightShiftScalar(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs,
                    out: desc.out.to_relative(converter),
                })
            }
        }
    }
}

impl RelativeOps for CustomOpRepr {
    fn to_relative(&self, converter: &mut OperationConverter) -> CustomOpRepr {
        let id = self.id.clone();

        CustomOpRepr {
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

impl<E: Element> RelativeOpsScalar<E> for NumericOperationRepr<E> {
    fn to_relative<F>(&self, converter: &mut OperationConverter, local_elem: F) -> Self
    where
        F: Fn(&mut OperationConverter, &E) -> E,
    {
        match self {
            NumericOperationRepr::Add(desc) => NumericOperationRepr::Add(BinaryOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::AddScalar(desc) => {
                NumericOperationRepr::AddScalar(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::Sub(desc) => NumericOperationRepr::Sub(BinaryOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::SubScalar(desc) => {
                NumericOperationRepr::SubScalar(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::Div(desc) => NumericOperationRepr::Div(BinaryOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::DivScalar(desc) => {
                NumericOperationRepr::DivScalar(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::Rem(desc) => NumericOperationRepr::Rem(BinaryOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::RemScalar(desc) => {
                NumericOperationRepr::RemScalar(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::Mul(desc) => NumericOperationRepr::Mul(BinaryOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::MulScalar(desc) => {
                NumericOperationRepr::MulScalar(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::Abs(desc) => NumericOperationRepr::Abs(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::Ones(desc) => {
                NumericOperationRepr::Ones(desc.to_relative(converter))
            }
            NumericOperationRepr::Zeros(desc) => {
                NumericOperationRepr::Zeros(desc.to_relative(converter))
            }
            NumericOperationRepr::Full(desc) => NumericOperationRepr::Full((
                desc.0.to_relative(converter),
                local_elem(converter, &desc.1),
            )),
            NumericOperationRepr::Gather(desc) => NumericOperationRepr::Gather(GatherOpRepr {
                tensor: desc.tensor.to_relative(converter),
                dim: desc.dim,
                indices: desc.indices.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::Scatter(desc) => NumericOperationRepr::Scatter(ScatterOpRepr {
                tensor: desc.tensor.to_relative(converter),
                dim: desc.dim,
                indices: desc.indices.to_relative(converter),
                value: desc.value.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::Select(desc) => NumericOperationRepr::Select(SelectOpRepr {
                tensor: desc.tensor.to_relative(converter),
                dim: desc.dim,
                indices: desc.indices.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::SelectAssign(desc) => {
                NumericOperationRepr::SelectAssign(SelectAssignOpRepr {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    indices: desc.indices.to_relative(converter),
                    value: desc.value.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::MaskWhere(desc) => {
                NumericOperationRepr::MaskWhere(MaskWhereOpRepr {
                    tensor: desc.tensor.to_relative(converter),
                    mask: desc.mask.to_relative(converter),
                    value: desc.value.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::MaskFill(desc) => {
                NumericOperationRepr::MaskFill(MaskFillOpRepr {
                    tensor: desc.tensor.to_relative(converter),
                    mask: desc.mask.to_relative(converter),
                    value: local_elem(converter, &desc.value),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::MeanDim(desc) => {
                NumericOperationRepr::MeanDim(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs, // Dim should stay the same.
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::Mean(desc) => NumericOperationRepr::Mean(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::Sum(desc) => NumericOperationRepr::Sum(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::SumDim(desc) => {
                NumericOperationRepr::SumDim(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs, // Dim should stay the same.
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::Prod(desc) => NumericOperationRepr::Prod(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::ProdDim(desc) => {
                NumericOperationRepr::ProdDim(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs, // Dim should stay the same.
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::EqualElem(desc) => {
                NumericOperationRepr::EqualElem(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::Greater(desc) => NumericOperationRepr::Greater(BinaryOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::GreaterElem(desc) => {
                NumericOperationRepr::GreaterElem(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::GreaterEqual(desc) => {
                NumericOperationRepr::GreaterEqual(BinaryOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::GreaterEqualElem(desc) => {
                NumericOperationRepr::GreaterEqualElem(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::Lower(desc) => NumericOperationRepr::Lower(BinaryOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::LowerElem(desc) => {
                NumericOperationRepr::LowerElem(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::LowerEqual(desc) => {
                NumericOperationRepr::LowerEqual(BinaryOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: desc.rhs.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::LowerEqualElem(desc) => {
                NumericOperationRepr::LowerEqualElem(ScalarOpRepr {
                    lhs: desc.lhs.to_relative(converter),
                    rhs: local_elem(converter, &desc.rhs),
                    out: desc.out.to_relative(converter),
                })
            }
            NumericOperationRepr::ArgMax(desc) => NumericOperationRepr::ArgMax(ScalarOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs,
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::ArgMin(desc) => NumericOperationRepr::ArgMin(ScalarOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs,
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::Max(desc) => NumericOperationRepr::Max(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::MaxDimWithIndices(desc) => {
                NumericOperationRepr::MaxDimWithIndices(ReduceDimWithIndicesOpRepr {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    out: desc.out.to_relative(converter),
                    out_indices: desc.out_indices.to_relative(converter),
                })
            }
            NumericOperationRepr::MinDimWithIndices(desc) => {
                NumericOperationRepr::MinDimWithIndices(ReduceDimWithIndicesOpRepr {
                    tensor: desc.tensor.to_relative(converter),
                    dim: desc.dim,
                    out: desc.out.to_relative(converter),
                    out_indices: desc.out_indices.to_relative(converter),
                })
            }
            NumericOperationRepr::Min(desc) => NumericOperationRepr::Min(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::MaxDim(desc) => NumericOperationRepr::MaxDim(ScalarOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs,
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::MinDim(desc) => NumericOperationRepr::MinDim(ScalarOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs,
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::Clamp(desc) => NumericOperationRepr::Clamp(ClampOpRepr {
                tensor: desc.tensor.to_relative(converter),
                min: local_elem(converter, &desc.min),
                max: local_elem(converter, &desc.max),
                out: desc.out.to_relative(converter),
            }),
            NumericOperationRepr::IntRandom(desc) => {
                NumericOperationRepr::IntRandom(RandomOpRepr {
                    out: desc.out.to_relative(converter),
                    distribution: desc.distribution,
                })
            }
            NumericOperationRepr::Powf(desc) => NumericOperationRepr::Powf(BinaryOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
        }
    }
}

impl RelativeOps for BaseOperationRepr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        match self {
            BaseOperationRepr::ToDevice(desc) => {
                BaseOperationRepr::ToDevice(desc.to_relative(converter))
            }
            BaseOperationRepr::Reshape(desc) => BaseOperationRepr::Reshape(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationRepr::SwapDims(desc) => BaseOperationRepr::SwapDims(SwapDimsOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
                dim1: desc.dim1,
                dim2: desc.dim2,
            }),
            BaseOperationRepr::Permute(desc) => BaseOperationRepr::Permute(PermuteOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
                axes: desc.axes.clone(),
            }),
            BaseOperationRepr::Expand(desc) => BaseOperationRepr::Expand(ExpandOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
                shape: desc.shape.clone(),
            }),
            BaseOperationRepr::Flip(desc) => BaseOperationRepr::Flip(FlipOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
                axes: desc.axes.clone(),
            }),
            BaseOperationRepr::Slice(desc) => BaseOperationRepr::Slice(SliceOpRepr {
                tensor: desc.tensor.to_relative(converter),
                ranges: desc.ranges.iter().map(|_range| 0..1).collect(),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationRepr::SliceAssign(desc) => {
                BaseOperationRepr::SliceAssign(SliceAssignOpRepr {
                    tensor: desc.tensor.to_relative(converter),
                    ranges: desc.ranges.iter().map(|_range| 0..1).collect(),
                    value: desc.value.to_relative(converter),
                    out: desc.out.to_relative(converter),
                })
            }
            BaseOperationRepr::Equal(desc) => BaseOperationRepr::Equal(BinaryOpRepr {
                lhs: desc.lhs.to_relative(converter),
                rhs: desc.rhs.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationRepr::RepeatDim(desc) => BaseOperationRepr::RepeatDim(RepeatDimOpRepr {
                tensor: desc.tensor.to_relative(converter),
                dim: desc.dim,
                times: desc.times,
                out: desc.out.to_relative(converter),
            }),
            BaseOperationRepr::Cat(desc) => BaseOperationRepr::Cat(CatOpRepr {
                tensors: desc
                    .tensors
                    .iter()
                    .map(|tensor| tensor.to_relative(converter))
                    .collect(),
                dim: desc.dim,
                out: desc.out.to_relative(converter),
            }),
            BaseOperationRepr::Cast(desc) => BaseOperationRepr::Cast(UnaryOpRepr {
                input: desc.input.to_relative(converter),
                out: desc.out.to_relative(converter),
            }),
            BaseOperationRepr::Empty(desc) => BaseOperationRepr::Empty(desc.to_relative(converter)),
        }
    }
}

impl RelativeOps for InitOperationRepr {
    fn to_relative(&self, converter: &mut OperationConverter) -> Self {
        Self {
            out: self.out.to_relative(converter),
        }
    }
}

impl RelativeOps for TensorRepr {
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
        let relative_tensor = TensorRepr {
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
    use burn_ir::{TensorId, TensorRepr, TensorStatus};
    use burn_tensor::DType;

    #[test]
    fn tensor_description_to_relative() {
        let tensor1 = TensorRepr {
            id: TensorId::new(500),
            shape: vec![512, 32, 2048],
            status: TensorStatus::ReadOnly,
            dtype: DType::F32,
        };
        let tensor2 = TensorRepr {
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
            TensorRepr {
                id: TensorId::new(0),
                shape: vec![1, 2, 3],
                status: TensorStatus::ReadOnly,
                dtype: DType::F32
            }
        );
        assert_eq!(
            tensor2_local,
            TensorRepr {
                id: TensorId::new(1),
                shape: vec![1, 4, 3],
                status: TensorStatus::ReadOnly,
                dtype: DType::F32
            }
        );
    }
}
