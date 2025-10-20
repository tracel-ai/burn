use burn_tensor::{
    DType, Device, Shape, TensorData, TensorPrimitive,
    ops::{FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{
        QParamTensor, QTensorPrimitive, QuantLevel, QuantMode, QuantParam, QuantPropagation,
        QuantScheme, QuantValue, QuantizationParametersPrimitive, params_shape,
    },
};
use cubecl::{
    matmul::components::{AccG, AccS, LhsG, LhsS, RhsG, RhsS},
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
fn new_qtensor<R: CubeRuntime>(
    data: &[u8],
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
) -> CubeTensor<R> {
    new_quantized(shape, scheme, device, Some(data))
}

/// Create an empty quantized tensor.
pub fn empty_qtensor<R: CubeRuntime>(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
) -> CubeTensor<R> {
    new_quantized(shape, scheme, device, None)
}

fn new_quantized<R: CubeRuntime>(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
    data: Option<&[u8]>,
) -> CubeTensor<R> {
    let client = R::client(device);
    let shape: Shape = shape.into();
    let mut shape_value: Shape = shape.clone();

    let rank = shape.rank();
    let shape_last = shape[rank - 1];
    let num_quants = scheme.num_quants();

    let data_size = match scheme.store {
        QuantStore::U32 => {
            if !shape_last.is_multiple_of(num_quants) {
                panic!("shape must be aligned to storage packing")
            }
            shape_value.dims[rank - 1] = shape_last.div_ceil(num_quants);
            size_of::<u32>()
        }
        QuantStore::Native => match scheme.value {
            QuantValue::Q8F | QuantValue::Q8S | QuantValue::E4M3 | QuantValue::E5M2 => {
                size_of::<i8>()
            }
            // Native e2m1 is packed in u8
            QuantValue::E2M1 => size_of::<u8>(),
            QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S => {
                panic!("Can't store native sub-byte values")
            }
        },
    };

    let scales_dtype = match scheme.param {
        QuantParam::F32 => DType::F32,
        QuantParam::F16 => DType::F16,
        QuantParam::BF16 => DType::BF16,
        // Represented by U8 and reinterpreted in the kernel
        QuantParam::UE8M0 | QuantParam::UE4M3 => DType::U8,
    };

    let scales_shape = params_shape(&shape, scheme.level);
    let data_desc = AllocationDescriptor::optimized(&shape_value.dims, data_size);
    let scales_desc = AllocationDescriptor::optimized(&scales_shape.dims, scales_dtype.size());

    let mut tensors = match data {
        Some(data) => {
            let num_bytes = shape_value.num_elements() * data_size;
            client.create_tensors(vec![
                (data_desc, &data[..num_bytes]),
                (scales_desc, &data[num_bytes..]),
            ])
        }
        None => client.empty_tensors(vec![data_desc, scales_desc]),
    };
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
                    value:
                        QuantValue::Q8F
                        | QuantValue::Q8S
                        | QuantValue::Q4F
                        | QuantValue::Q4S
                        | QuantValue::Q2F
                        | QuantValue::Q2S
                        | QuantValue::E4M3
                        | QuantValue::E5M2
                        | QuantValue::E2M1,
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
                QuantValue::Q8F | QuantValue::Q8S => into_data::<R, i8>(values).await,
                QuantValue::E4M3 | QuantValue::E5M2 | QuantValue::E2M1 => {
                    into_data::<R, u8>(values).await
                }
                QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S => {
                    panic!("Can't store native sub-byte values")
                }
            },
            QuantStore::U32 => into_data::<R, u32>(values).await,
        };
        let data_params = match scheme.param {
            QuantParam::UE8M0 | QuantParam::UE4M3 => into_data::<R, u8>(params).await,
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

    fn q_slice(
        _tensor: QuantizedTensor<Self>,
        _slices: &[burn_tensor::Slice],
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_expand(_tensor: QuantizedTensor<Self>, _shape: Shape) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        let (propagation, scheme) = match (&lhs, &rhs) {
            (TensorPrimitive::QFloat(lhs), _) => (lhs.propagation(), *lhs.scheme()),
            (_, TensorPrimitive::QFloat(rhs)) => (rhs.propagation(), *rhs.scheme()),
            _ => unreachable!(),
        };

        // Inherit precision for mixed inputs, default to `FloatElem` for fully quantized.
        let out_dtype = match (&lhs, &rhs) {
            (TensorPrimitive::Float(lhs), _) => lhs.dtype,
            (_, TensorPrimitive::Float(rhs)) => rhs.dtype,
            _ => F::dtype(),
        };

        let (lhs_dtype, lhs) = match lhs {
            TensorPrimitive::Float(lhs) => (lhs.dtype, lhs),
            TensorPrimitive::QFloat(lhs) => (out_dtype, lhs),
        };
        let (rhs_dtype, rhs) = match rhs {
            TensorPrimitive::Float(rhs) => (rhs.dtype, rhs),
            TensorPrimitive::QFloat(rhs) => (out_dtype, rhs),
        };

        let out = execute_with_dtype!(float(lhs_dtype), LP, {
            execute_with_dtype!(float(rhs_dtype), RP, {
                execute_with_dtype!(float(out_dtype), OP, {
                    type MP = (LhsG<LP>, RhsG<RP>, AccG<OP>, LhsS<LP>, RhsS<RP>, AccS<OP>);

                    kernel::matmul::matmul::<R, MP>(lhs, rhs, None, MatmulStrategy::default())
                        .unwrap()
                })
            })
        });

        match propagation {
            QuantPropagation::Propagate => {
                TensorPrimitive::QFloat(Self::quantize_dynamic(out, &scheme))
            }
            QuantPropagation::Inhibit => TensorPrimitive::Float(out),
        }
    }
}
