use burn_backend::{
    Bytes, DType, ExecutionError, Shape, SplitPolicy, TensorData, TensorMetadata, TensorPrimitive,
    get_device_settings,
    ops::QTensorOps,
    quantization::{
        QParamTensor, QuantMode, QuantPropagation, QuantScheme, QuantValue,
        QuantizationParametersPrimitive, ScaleDtype, global_scale_dtype, params_shape,
    },
    tensor::{Device, FloatTensor, QuantizedTensor},
};
use burn_std::{FloatDType, Metadata, quantization::global_scale_size};
use cubecl::server::{MemoryLayout, MemoryLayoutDescriptor, MemoryLayoutStrategy};
use cubecl::{e2m1x2, quant::scheme::QuantStore};

use crate::{
    CubeBackend, CubeRuntime,
    kernel::{self, matmul::MatmulStrategy},
    tensor::{CubeTensor, QParams},
};

use super::{into_data, permute, swap_dims};

/// Length of the block-scales region within a combined scales+global byte buffer.
fn scales_region_len(total: usize, scheme: &QuantScheme) -> usize {
    total
        .checked_sub(global_scale_size(scheme))
        .expect("quantized tensor data is shorter than the scheme's global scale")
}

/// Create a quantized tensor with packed values (u32).
fn new_qtensor_optimized<R: CubeRuntime>(
    data: Bytes,
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
) -> CubeTensor<R> {
    new_qtensor(data, shape, scheme, device, MemoryLayoutStrategy::Optimized)
}

/// Create a quantized tensor with packed values (u32).
fn new_qtensor<R: CubeRuntime>(
    data: Bytes,
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
    kind: MemoryLayoutStrategy,
) -> CubeTensor<R> {
    new_quantized(shape, scheme, device, Some(data), kind)
}

/// Create an empty quantized tensor.
pub fn empty_qtensor_optimized<R: CubeRuntime>(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
) -> CubeTensor<R> {
    empty_qtensor(shape, scheme, device, MemoryLayoutStrategy::Optimized)
}

/// Create an empty quantized tensor.
pub fn empty_qtensor<R: CubeRuntime>(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
    kind: MemoryLayoutStrategy,
) -> CubeTensor<R> {
    new_quantized(shape, scheme, device, None, kind)
}

fn new_quantized<R: CubeRuntime>(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
    data: Option<Bytes>,
    alloc_kind: MemoryLayoutStrategy,
) -> CubeTensor<R> {
    let client = R::client(device);
    let shape: Shape = shape.into();
    let mut shape_value: Shape = shape.clone();

    let rank = shape.rank();
    let shape_last = shape[rank - 1];
    let num_quants = scheme.num_quants();

    let data_size = match scheme.store {
        QuantStore::PackedU32(_) => {
            if !shape_last.is_multiple_of(num_quants) {
                panic!("Can't store in u32")
            }
            shape_value[rank - 1] = shape_last.div_ceil(num_quants);
            size_of::<u32>()
        }
        QuantStore::Native => match scheme.value {
            QuantValue::Q8F | QuantValue::Q8S | QuantValue::E4M3 | QuantValue::E5M2 => {
                size_of::<i8>()
            }
            QuantValue::Q4F
            | QuantValue::Q4S
            | QuantValue::Q2F
            | QuantValue::Q2S
            | QuantValue::E2M1 => {
                panic!("Can't store native sub-byte values")
            }
        },
        QuantStore::PackedNative(_) => match scheme.value {
            QuantValue::E2M1 => size_of::<e2m1x2>(),
            other => panic!("{other:?} doesn't support native packing"),
        },
    };

    let scales_dtype = match scheme.scale_dtype() {
        ScaleDtype::F32 => DType::F32,
        ScaleDtype::F16 => DType::F16,
        ScaleDtype::BF16 => DType::BF16,
        // Represented by U8 and reinterpreted in the kernel
        ScaleDtype::UE8M0 | ScaleDtype::UE4M3 => DType::U8,
    };

    let scales_shape = params_shape(&shape, &scheme);
    let data_desc = MemoryLayoutDescriptor::new(alloc_kind, shape_value.clone(), data_size);
    let scales_desc =
        MemoryLayoutDescriptor::new(alloc_kind, scales_shape.clone(), scales_dtype.size());

    let global_shape = Shape::new([1]);
    let global_dtype = global_scale_dtype(&scheme).map(|dtype| {
        // The region is f32-sized and the kernels bind it as f32.
        assert_eq!(
            dtype,
            ScaleDtype::F32,
            "a two-level scheme binds its per-tensor scale as f32, got {scheme:?}"
        );
        DType::F32
    });
    let global_desc = global_dtype
        .map(|dtype| MemoryLayoutDescriptor::new(alloc_kind, global_shape.clone(), dtype.size()));

    let mut tensors = match data {
        Some(data) => {
            let num_bytes = shape_value.num_elements() * data_size;
            let split = data.split(num_bytes, SplitPolicy::Shared);

            match (split, global_desc.clone()) {
                (Ok((bytes_data, bytes_params)), None) => client
                    .create_tensors(vec![(data_desc, bytes_data), (scales_desc, bytes_params)]),
                (Ok((bytes_data, bytes_params)), Some(global_desc)) => {
                    let scales_bytes = scales_region_len(bytes_params.len(), &scheme);
                    match bytes_params.split(scales_bytes, SplitPolicy::Shared) {
                        Ok((block, global)) => client.create_tensors(vec![
                            (data_desc, bytes_data),
                            (scales_desc, block),
                            (global_desc, global),
                        ]),
                        Err((params, _)) => client.create_tensors_from_slices(vec![
                            (data_desc, &bytes_data[..]),
                            (scales_desc, &params[..scales_bytes]),
                            (global_desc, &params[scales_bytes..]),
                        ]),
                    }
                }
                (Err((data, _)), global_desc) => {
                    let params = &data[num_bytes..];
                    let scales_bytes = scales_region_len(params.len(), &scheme);
                    let mut entries = vec![
                        (data_desc, &data[..num_bytes]),
                        (scales_desc, &params[..scales_bytes]),
                    ];
                    if let Some(global_desc) = global_desc {
                        entries.push((global_desc, &params[scales_bytes..]));
                    }
                    client.create_tensors_from_slices(entries)
                }
            }
        }
        None => {
            let mut descs = vec![data_desc, scales_desc];
            descs.extend(global_desc);
            client.empty_tensors(descs)
        }
    };

    let global = global_dtype.map(|dtype| {
        let MemoryLayout {
            memory: handle,
            strides,
        } = tensors.remove(2);
        QParamTensor {
            offset_start: handle.offset_start.unwrap_or(0) as usize,
            offset_end: handle.offset_end.unwrap_or(0) as usize,
            metadata: Metadata::new(global_shape, strides),
            dtype,
        }
    });
    let MemoryLayout {
        memory: scales_handle,
        strides: scales_strides,
    } = tensors.remove(1);
    let MemoryLayout { memory, strides } = tensors.remove(0);

    let scales = QParamTensor {
        offset_start: scales_handle.offset_start.unwrap_or(0) as usize,
        offset_end: scales_handle.offset_end.unwrap_or(0) as usize,
        metadata: Metadata::new(scales_shape, scales_strides),
        dtype: scales_dtype,
    };
    let qparams = QParams { scales, global };

    CubeTensor::new_quantized(
        client,
        memory,
        shape,
        device.clone(),
        strides,
        DType::QFloat(scheme),
        qparams,
    )
}

impl<R: CubeRuntime> QTensorOps<Self> for CubeBackend<R> {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        match data.dtype {
            DType::QFloat(scheme) => match scheme {
                QuantScheme {
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
                    new_qtensor_optimized(data.bytes, data.shape.clone(), scheme, device)
                }
                QuantScheme {
                    mode: QuantMode::Lookup,
                    ..
                } => unimplemented!("lookup quantization does not travel as a QFloat tensor"),
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
        // The kernel reads this at the scheme's scale dtype, not the tensor's actual dtype.
        if let Some(global) = &qparams.global {
            assert_eq!(
                global.dtype,
                DType::F32,
                "a two-level scheme's per-tensor scale must be an f32 tensor, got {:?}",
                global.dtype
            );
        }
        kernel::quantization::quantize(tensor, scheme, qparams.scales, qparams.global)
    }

    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        kernel::quantization::dequantize(tensor, dtype.into())
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        super::to_device(tensor, device)
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        super::q_reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        if tensor.qparams.is_none() {
            return into_data(tensor).await;
        }

        let (shape, dtype) = (tensor.shape(), tensor.dtype);
        let global = tensor.global();
        let (values, params) = tensor.quantized_handles().unwrap();

        let mut data_values = into_data(values).await?;
        let data_params = into_data(params).await?;

        data_values.bytes.extend_from_byte_slice(&data_params.bytes);

        if let Some(global) = global {
            let data_global = into_data(global).await?;
            data_values.bytes.extend_from_byte_slice(&data_global.bytes);
        }

        Ok(TensorData {
            bytes: data_values.bytes,
            shape,
            dtype,
        })
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

    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        let (settings, scheme) = match (&lhs, &rhs) {
            (TensorPrimitive::QFloat(lhs), _) => {
                (get_device_settings::<Self>(&lhs.device), lhs.scheme())
            }
            (_, TensorPrimitive::QFloat(rhs)) => {
                (get_device_settings::<Self>(&rhs.device), rhs.scheme())
            }
            _ => unreachable!(),
        };

        // Inherit precision for mixed inputs, default to `FloatElem` for fully quantized.
        let out_dtype = match (&lhs, &rhs) {
            (TensorPrimitive::Float(lhs), _) => lhs.dtype,
            (_, TensorPrimitive::Float(rhs)) => rhs.dtype,
            _ => settings.float_dtype.into(),
        };

        let (_lhs_dtype, lhs) = match lhs {
            TensorPrimitive::Float(lhs) => (lhs.dtype, lhs),
            TensorPrimitive::QFloat(lhs) => (out_dtype, lhs),
        };
        let (_rhs_dtype, rhs) = match rhs {
            TensorPrimitive::Float(rhs) => (rhs.dtype, rhs),
            TensorPrimitive::QFloat(rhs) => (out_dtype, rhs),
        };

        let out =
            kernel::matmul::matmul(lhs, rhs, None, MatmulStrategy::default(), out_dtype).unwrap();

        match settings.quantization.propagation {
            QuantPropagation::Propagate => {
                TensorPrimitive::QFloat(Self::quantize_dynamic(out, &scheme))
            }
            QuantPropagation::Inhibit => TensorPrimitive::Float(out),
        }
    }
}
