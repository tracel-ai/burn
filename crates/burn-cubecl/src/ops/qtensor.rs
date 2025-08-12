use std::ops::Range;

use burn_tensor::{
    DType, Device, Shape, TensorData, TensorPrimitive,
    ops::{FloatTensor, FloatTensorOps, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{
        QParamTensor, QTensorPrimitive, QuantLevel, QuantMode, QuantParam, QuantPropagation,
        QuantScheme, QuantValue, QuantizationParametersPrimitive,
    },
};
use cubecl::{
    Feature, Runtime,
    client::ComputeClient,
    ir::{Elem, IntKind},
    server::{Allocation, AllocationDescriptor},
};
use cubecl_quant::scheme::QuantStore;

use crate::{
    CubeBackend, CubeRuntime, FloatElement, IntElement,
    element::BoolElement,
    execute_with_dtype,
    kernel::{self, matmul::MatmulStrategy},
    tensor::{CubeTensor, QParams},
};

use super::{into_data, permute, swap_dims};

/// Create a quantized tensor with packed values (u32).
fn new_qtensor<R: CubeRuntime, S: Into<Shape>>(
    data: &[u8],
    shape: S,
    scheme: QuantScheme,
    device: &R::Device,
) -> CubeTensor<R> {
    let client = R::client(device);
    let shape: Shape = shape.into();
    let scales_shape: Shape;
    let scales_dtype = match scheme.param {
        QuantParam::F32 => DType::F32,
        QuantParam::F16 => DType::F16,
        QuantParam::BF16 => DType::BF16,
    };

    let descriptors = match scheme {
        // Just to ensure we get and error if more modes are added and unhandled
        QuantScheme {
            level: QuantLevel::Tensor,
            mode: QuantMode::Symmetric,
            value: QuantValue::QInt8,
            ..
        } => {
            let data_desc = AllocationDescriptor::optimized(&shape.dims, size_of::<i8>());
            let scale_desc = AllocationDescriptor::optimized(&[1], scales_dtype.size());

            scales_shape = Shape::new([1]);
            vec![
                (data_desc, &data[..shape.num_elements()]),
                (scale_desc, &data[shape.num_elements()..]),
            ]
        }
        QuantScheme {
            level: QuantLevel::Block(block_size),
            mode: QuantMode::Symmetric,
            value: QuantValue::QInt8,
            ..
        } => {
            let numel = shape.num_elements();
            let num_blocks = numel / block_size;
            scales_shape = Shape::new([num_blocks]);
            let data_desc = AllocationDescriptor::optimized(&shape.dims, size_of::<i8>());
            let scales_desc =
                AllocationDescriptor::optimized(&scales_shape.dims, scales_dtype.size());
            vec![(data_desc, &data[..numel]), (scales_desc, &data[numel..])]
        }
    };

    let mut tensors = client.create_tensors(descriptors);
    let Allocation {
        handle: scales_handle,
        strides: scales_strides,
    } = tensors.remove(1);
    let Allocation { handle, strides } = tensors.remove(0);

    let scales = QParamTensor {
        offset_start: scales_handle.offset_start.unwrap_or(0) as usize,
        offset_end: scales_handle.offset_end.unwrap_or(0) as usize,
        shape: scales_shape,
        strides: scales_strides,
        dtype: scales_dtype,
    };
    let qparams = QParams { scales };

    CubeTensor::new_quantized(
        client,
        handle,
        shape,
        device.clone(),
        strides,
        DType::QFloat(scheme),
        qparams,
    )
}

/// Create an empty quantized tensor.
pub fn empty_qtensor<R: CubeRuntime>(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
) -> CubeTensor<R> {
    let client = R::client(device);
    let shape: Shape = shape.into();
    let mut shape_value: Shape = shape.clone();

    let scales_shape: Shape;
    let rank = shape.dims.len();
    let shape_last = shape.dims[rank - 1];
    let num_quants = scheme.num_quants();

    let data_size = match scheme.store {
        QuantStore::U32 => {
            if shape_last % num_quants != 0 {
                panic!("Can't store in u32, padding not yet implemented for quantization.");
            }
            shape_value.dims[rank - 1] = shape_last / num_quants;
            size_of::<u32>()
        }
        QuantStore::Native => match scheme.value {
            QuantValue::QInt8 => size_of::<i8>(),
        },
    };

    let scales_dtype = match scheme.param {
        QuantParam::F32 => DType::F32,
        QuantParam::F16 => DType::F16,
        QuantParam::BF16 => DType::BF16,
    };
    let descriptors = match scheme {
        // Just to ensure we get and error if more modes are added and unhandled
        QuantScheme {
            level: QuantLevel::Tensor,
            mode: QuantMode::Symmetric,
            value: QuantValue::QInt8,
            ..
        } => {
            let data_desc = AllocationDescriptor::contiguous(&shape_value.dims, data_size);
            let scale_desc = AllocationDescriptor::contiguous(&[1], scales_dtype.size());
            scales_shape = Shape::new([1]);
            vec![data_desc, scale_desc]
        }
        QuantScheme {
            level: QuantLevel::Block(block_size),
            mode: QuantMode::Symmetric,
            value: QuantValue::QInt8,
            ..
        } => {
            let num_blocks = shape.num_elements() / block_size;
            scales_shape = Shape::new([num_blocks]);
            let data_desc = AllocationDescriptor::contiguous(&shape_value.dims, data_size);
            let scales_desc =
                AllocationDescriptor::contiguous(&scales_shape.dims, scales_dtype.size());
            vec![data_desc, scales_desc]
        }
    };

    let mut tensors = client.empty_tensors(descriptors);
    let Allocation {
        handle: scales_handle,
        strides: scales_strides,
    } = tensors.remove(1);
    let Allocation { handle, strides } = tensors.remove(0);

    let scales = QParamTensor {
        offset_start: scales_handle.offset_start.unwrap_or(0) as usize,
        offset_end: scales_handle.offset_end.unwrap_or(0) as usize,
        shape: scales_shape,
        strides: scales_strides,
        dtype: scales_dtype,
    };
    let qparams = QParams { scales };

    CubeTensor::new_quantized(
        client,
        handle,
        shape,
        device.clone(),
        strides,
        DType::QFloat(scheme),
        qparams,
    )
}

impl<R, F, I, BT> QTensorOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        match data.dtype {
            DType::QFloat(scheme) => match scheme {
                QuantScheme {
                    level: QuantLevel::Tensor | QuantLevel::Block(_),
                    mode: QuantMode::Symmetric,
                    value: QuantValue::QInt8,
                    ..
                } => {
                    // TensorData quantized representation is the same, with multiple quantized values
                    // packed into u32 and quantization parameters appended to the bytes
                    new_qtensor(data.as_bytes(), data.shape.clone(), scheme, device)
                }
            },
            _ => panic!(
                "Invalid dtype (expected DType::QFloat, got {:?})",
                data.dtype
            ),
        }
    }

    // TODO: quantize_dynamic (we can compute min-max on the fly and scale, especially when not per-tensor)

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        kernel::quantization::quantize::<R, F>(tensor, scheme, qparams.scales)
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        kernel::quantization::dequantize::<R, F>(tensor)
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        super::to_device(tensor, device)
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        super::reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> TensorData {
        if tensor.qparams.is_none() {
            return execute_with_dtype!(tensor.dtype, E, into_data::<R, E>(tensor).await);
        }

        let (shape, dtype) = (tensor.shape.dims.clone(), tensor.dtype);
        let scheme = match dtype {
            DType::QFloat(val) => val,
            _ => unreachable!("Already checked if quantized."),
        };
        let (values, params) = tensor.quantized_handles().unwrap();

        let mut data_values = match scheme.store {
            QuantStore::Native => match scheme.value {
                QuantValue::QInt8 => into_data::<R, i8>(values).await,
            },
            QuantStore::U32 => into_data::<R, u32>(values).await,
        };
        let data_params = match scheme.param {
            QuantParam::F16 => into_data::<R, half::f16>(params).await,
            QuantParam::BF16 => into_data::<R, half::bf16>(params).await,
            QuantParam::F32 => into_data::<R, f32>(params).await,
        };

        data_values.bytes.extend_from_byte_slice(&data_params.bytes);

        TensorData {
            bytes: data_values.bytes,
            shape,
            dtype,
        }
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        permute(tensor, axes)
    }

    fn q_flip(_tensor: QuantizedTensor<Self>, _axes: &[usize]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_gather(
        _dim: usize,
        _tensor: QuantizedTensor<Self>,
        _indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_select(
        _tensor: QuantizedTensor<Self>,
        _dim: usize,
        _indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_slice(_tensor: QuantizedTensor<Self>, _ranges: &[Range<usize>]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_expand(_tensor: QuantizedTensor<Self>, _shape: Shape) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_matmul(lhs: QuantizedTensor<Self>, rhs: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        if features_enabled::<R>(&lhs.client)
            && both_matches_symmetric_qint8(lhs.scheme(), rhs.scheme())
        {
            let out =
                kernel::matmul::q_matmul(lhs.clone(), rhs.clone(), None, MatmulStrategy::default());
            if let Ok(out) = out {
                return match lhs.propagation() {
                    QuantPropagation::Propagate => {
                        TensorPrimitive::QFloat(Self::quantize_dynamic(out, lhs.scheme()))
                    }
                    QuantPropagation::Inhibit => TensorPrimitive::Float(out),
                };
            }
        }

        // If the above quantized matmul fail, we fallback to the dequantize-then-matmul pattern.
        let scheme = *lhs.scheme();
        let propagation = lhs.propagation();
        let t1_f = <Self>::dequantize(lhs);
        let t2_f = <Self>::dequantize(rhs);
        let out = Self::float_matmul(t1_f, t2_f);

        match propagation {
            QuantPropagation::Propagate => {
                TensorPrimitive::QFloat(Self::quantize_dynamic(out, &scheme))
            }
            QuantPropagation::Inhibit => TensorPrimitive::Float(out),
        }
    }
}

fn both_matches_symmetric_qint8(lhs: &QuantScheme, rhs: &QuantScheme) -> bool {
    [lhs, rhs].iter().all(|scheme| {
        matches!(
            scheme,
            QuantScheme {
                level: QuantLevel::Tensor,
                mode: QuantMode::Symmetric,
                value: QuantValue::QInt8,
                ..
            }
        )
    })
}

fn features_enabled<R: Runtime>(client: &ComputeClient<R::Server, R::Channel>) -> bool {
    client
        .properties()
        .feature_enabled(Feature::Type(Elem::Int(IntKind::I8)))
        && client
            .properties()
            .feature_enabled(Feature::DynamicLineSize)
}
