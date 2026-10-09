use burn_backend::{
    Bytes, DType, ExecutionError, Shape, SplitPolicy, TensorData, TensorMetadata, TensorPrimitive,
    get_or_init_device_settings,
    ops::QTensorOps,
    quantization::{
        QParamTensor, QuantMode, QuantPropagation, QuantScheme, QuantValue,
        QuantizationParametersPrimitive, ScaleDtype, global_scale_dtype, params_shape,
    },
    tensor::{Device, FloatTensor, QuantizedTensor},
};
use burn_std::{
    FloatDType, Metadata,
    quantization::{QPARAM_ALIGN, global_scale_size},
};
use cubecl::server::{MemoryLayout, MemoryLayoutDescriptor, MemoryLayoutStrategy};
use cubecl::{e2m1x2, quant::scheme::QuantStore};

use crate::{
    CubeBackend, CubeDevice,
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
fn new_qtensor_optimized(
    data: Bytes,
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &CubeDevice,
) -> CubeTensor {
    new_qtensor(data, shape, scheme, device, MemoryLayoutStrategy::Optimized)
}

/// Create a quantized tensor with packed values (u32).
fn new_qtensor(
    data: Bytes,
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &CubeDevice,
    kind: MemoryLayoutStrategy,
) -> CubeTensor {
    new_quantized(shape, scheme, device, Some(data), kind)
}

/// Create an empty quantized tensor.
pub fn empty_qtensor_optimized(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &CubeDevice,
) -> CubeTensor {
    empty_qtensor(shape, scheme, device, MemoryLayoutStrategy::Optimized)
}

/// Create an empty quantized tensor.
pub fn empty_qtensor(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &CubeDevice,
    kind: MemoryLayoutStrategy,
) -> CubeTensor {
    new_quantized(shape, scheme, device, None, kind)
}

/// The axis a packed `scheme` packs along on a tensor of `rank`, when it is not the innermost.
///
/// Such a tensor is stored with that axis swapped innermost, which is where packing puts its
/// words, and presented with the two swapped back: its bytes are the stored tensor's, row-major,
/// so a packed word never straddles two of the axes it does not pack.
fn outer_packed_axis(scheme: &QuantScheme, rank: usize) -> Option<usize> {
    match scheme.store {
        QuantStore::PackedU32(packed) | QuantStore::PackedNative(packed) if packed != 0 => {
            Some(rank - packed - 1)
        }
        _ => None,
    }
}

/// `scheme` as it reads on the tensor with `axis` swapped innermost: packed along the
/// innermost axis, its blocks swapped with it.
fn packed_innermost(mut scheme: QuantScheme, rank: usize, axis: usize) -> QuantScheme {
    scheme.store = match scheme.store {
        QuantStore::PackedU32(_) => QuantStore::PackedU32(0),
        QuantStore::PackedNative(_) => QuantStore::PackedNative(0),
        QuantStore::Native => QuantStore::Native,
    };
    if scheme.block_size().is_some() {
        scheme.swap_block_dims(rank, axis, rank - 1);
    }
    scheme
}

fn new_quantized(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &CubeDevice,
    data: Option<Bytes>,
    alloc_kind: MemoryLayoutStrategy,
) -> CubeTensor {
    let shape: Shape = shape.into();
    if let Some(axis) = outer_packed_axis(&scheme, shape.rank()) {
        let (rank, innermost) = (shape.rank(), shape.rank() - 1);
        let stored_shape = shape
            .swapped(axis, innermost)
            .expect("the packed axis is one of the tensor's");
        let stored = new_quantized(
            stored_shape,
            packed_innermost(scheme, rank, axis),
            device,
            data,
            alloc_kind,
        );
        return swap_dims(stored, axis, innermost);
    }

    let client = device.client();
    let mut shape_value: Shape = shape.clone();

    let rank = shape.rank();
    let shape_last = shape[rank - 1];
    let num_quants = scheme.num_quants();

    let data_size = match scheme.store {
        QuantStore::PackedU32(_) => {
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

/// Repack host-native Q4/Q2 values into the u32 layout used by CubeCL.
fn pack_subbyte_values(values: &[i8], shape: &Shape, bits: usize) -> Vec<u8> {
    let rank = shape.rank();
    let row_len = shape[rank - 1];
    if row_len == 0 {
        return Vec::new();
    }
    let values_per_word = u32::BITS as usize / bits;
    let words_per_row = row_len.div_ceil(values_per_word);
    let rows = values.len() / row_len;
    let mask = (1u32 << bits) - 1;
    let mut packed = Vec::with_capacity(rows * words_per_row * size_of::<u32>());

    for row in 0..rows {
        for word_index in 0..words_per_row {
            let mut word = 0u32;
            for value_index in 0..values_per_word {
                let column = word_index * values_per_word + value_index;
                if column < row_len {
                    let value = values[row * row_len + column] as u8 as u32;
                    word |= (value & mask) << (value_index * bits);
                }
            }
            packed.extend_from_slice(&word.to_ne_bytes());
        }
    }

    packed
}

fn q_from_native_subbyte_data(
    data: TensorData,
    scheme: QuantScheme,
    device: &CubeDevice,
) -> CubeTensor {
    let shape = data.shape().clone();
    let num_values = shape.num_elements();
    let raw = data.as_bytes();
    let values = raw[..num_values]
        .iter()
        .map(|&value| value as i8)
        .collect::<Vec<_>>();
    let qparams_start = num_values.div_ceil(QPARAM_ALIGN) * QPARAM_ALIGN;
    let packed_scheme = scheme.with_store(QuantStore::PackedU32(0));
    let bits = scheme.value.size_bits();
    let mut packed = pack_subbyte_values(&values, &shape, bits);
    packed.extend_from_slice(&raw[qparams_start..]);

    new_qtensor_optimized(Bytes::from_bytes_vec(packed), shape, packed_scheme, device)
}

impl QTensorOps<Self> for CubeBackend {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        let scheme = match data.dtype() {
            DType::QFloat(scheme) => scheme,
            _ => panic!(
                "Invalid dtype (expected DType::QFloat, got {:?})",
                data.dtype()
            ),
        };

        if scheme.mode == QuantMode::Lookup {
            unimplemented!("lookup quantization does not travel as a QFloat tensor");
        }

        if scheme.store == QuantStore::Native
            && matches!(
                scheme.value,
                QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S
            )
        {
            return q_from_native_subbyte_data(data, scheme, device);
        }

        let (bytes, shape, _) = data.into_parts();
        new_qtensor_optimized(bytes, shape, scheme, device)
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
        // Storage tiles are a layout for one machine's kernels, laid out at load; what is saved
        // is the rows every machine reads.
        assert!(
            !tensor.meta.is_tiled(),
            "q_into_data: a storage-tiled quantized tensor is not saved; save the weight it was \
             tiled from"
        );

        let (shape, dtype) = (tensor.shape(), tensor.dtype);
        // Written as stored, packed axis innermost — the bytes `q_from_data` reads back.
        let tensor = match outer_packed_axis(&tensor.scheme(), shape.rank()) {
            Some(axis) => swap_dims(tensor, axis, shape.rank() - 1),
            None => tensor,
        };
        let global = tensor.global();
        let (values, params) = tensor.quantized_handles().unwrap();

        let mut bytes = into_data(values).await?.into_bytes();
        let data_params = into_data(params).await?;

        bytes.extend_from_byte_slice(data_params.as_bytes());

        if let Some(global) = global {
            let data_global = into_data(global).await?;
            bytes.extend_from_byte_slice(data_global.as_bytes());
        }

        Ok(TensorData::from_bytes(bytes, shape, dtype))
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
            (TensorPrimitive::QFloat(lhs), _) => (
                get_or_init_device_settings::<Self>(&lhs.device),
                lhs.scheme(),
            ),
            (_, TensorPrimitive::QFloat(rhs)) => (
                get_or_init_device_settings::<Self>(&rhs.device),
                rhs.scheme(),
            ),
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

#[cfg(test)]
mod tests {
    use super::{CubeBackend, CubeDevice, pack_subbyte_values};
    use burn_backend::{
        DType, Shape, TensorData,
        ops::QTensorOps,
        quantization::{QuantScheme, QuantStore, QuantValue, QuantizedBytes},
    };

    #[test]
    fn pack_subbyte_values_pads_each_row_separately() {
        let packed = pack_subbyte_values(&[1, -2, 3, -4, 5, -6], &Shape::new([2, 3]), 4);

        assert_eq!(&packed[..4], &993u32.to_ne_bytes());
        assert_eq!(&packed[4..], &2652u32.to_ne_bytes());
    }

    #[test]
    fn native_q4_data_roundtrips_with_odd_rows() {
        let device = CubeDevice::default();
        let values = vec![1i8, -2, 3, -4, 5, -6];
        let data = TensorData::quantized(
            values.clone(),
            [2, 3],
            QuantScheme::default().with_value(QuantValue::Q4S),
            &[1.0],
            None,
        );

        assert!(
            matches!(data.dtype(), DType::QFloat(scheme) if scheme.store == QuantStore::Native)
        );
        let tensor = CubeBackend::q_from_data(data, &device);
        let restored = burn_std::future::block_on(CubeBackend::q_into_data(tensor)).unwrap();

        let (bytes, shape, dtype) = restored.into_parts();
        let DType::QFloat(scheme) = dtype else {
            panic!("Cube q_into_data must return a quantized dtype");
        };
        let (restored, _) = QuantizedBytes {
            bytes,
            shape,
            scheme,
        }
        .into_vec_i8();

        assert_eq!(restored, values);
    }
}
