use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, TensorMetadata};
use cubecl::quant::scheme::{QuantStore, QuantValue};
use cubecl::server::MemoryLayoutStrategy;

use crate::{ops::empty_qtensor, tensor::CubeTensor};

/// A storage-tiled tensor laid out in rows again, through cubek's unpack; a plain tensor as it
/// is. A tensor is packed for one matmul and read there through its binding; every other kernel
/// and every layout rewrite reads rows, so this is their head.
///
/// # Panics
///
/// A quantized storage-tiled tensor: nothing packs one.
pub fn untile(tensor: CubeTensor) -> CubeTensor {
    if !tensor.meta.is_tiled() {
        return tensor;
    }
    assert!(
        tensor.qparams.is_none(),
        "untile: a quantized tensor is never storage-tiled"
    );
    let (client, device, dtype) = (tensor.client.clone(), tensor.device.clone(), tensor.dtype);
    let output =
        cubek::matmul::tiled::pack::unpack(&client, tensor.binding(), dtype_to_storage_type(dtype))
            .expect("a storage-tiled binding describes its own tiles");
    CubeTensor::new(client, output.handle, *output.metadata, device, dtype)
}

/// Make a jit tensor contiguous.
pub fn into_contiguous(tensor: CubeTensor) -> CubeTensor {
    // A packed buffer has row-major strides over its physical dims, which is not rows.
    let tensor = untile(tensor);
    if tensor.is_contiguous() {
        return tensor;
    }

    if tensor.qparams.is_some() {
        return into_contiguous_quantized(tensor, MemoryLayoutStrategy::Contiguous);
    }

    let (client, device, dtype) = (tensor.client.clone(), tensor.device.clone(), tensor.dtype);

    let output = cubecl::std::tensor::into_contiguous(
        &client,
        tensor.binding(),
        dtype_to_storage_type(dtype),
    );

    CubeTensor::new(
        client.clone(),
        output.handle,
        *output.metadata,
        device,
        dtype,
    )
}

/// Make a jit tensor contiguous with an aligned last stride. Tensor is considered already contiguous
/// if runtime can read it as is. This is equivalent in practice.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(level = "trace", skip(tensor))
)]
pub fn into_contiguous_aligned(tensor: CubeTensor) -> CubeTensor {
    let tensor = untile(tensor);
    if tensor
        .device
        .can_read_tensor(tensor.meta.shape(), tensor.meta.strides())
    {
        return tensor;
    }

    if tensor.qparams.is_some() {
        return into_contiguous_quantized(tensor, MemoryLayoutStrategy::Optimized);
    }

    let (client, device, dtype) = (tensor.client.clone(), tensor.device.clone(), tensor.dtype);

    let output = cubecl::std::tensor::into_contiguous_pitched(
        &client,
        tensor.binding(),
        dtype_to_storage_type(dtype),
    );

    CubeTensor::new(
        client.clone(),
        output.handle,
        *output.metadata,
        device,
        dtype,
    )
}

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(level = "trace", skip(tensor))
)]
fn into_contiguous_quantized(tensor: CubeTensor, strategy: MemoryLayoutStrategy) -> CubeTensor {
    let scheme = tensor.scheme();
    let output = empty_qtensor(tensor.shape(), tensor.scheme(), &tensor.device, strategy);
    let (values, scales) = tensor.quantized_handles().unwrap();
    let (out_values, out_scales) = output.quantized_handles().unwrap();

    let (client, dtype_scales, dtype_value) = (scales.client.clone(), scales.dtype, values.dtype);

    match scheme.store {
        QuantStore::PackedU32(packed_dim) => {
            cubecl::std::tensor::into_contiguous_packed_ref(
                &client,
                values.binding(),
                out_values.binding(),
                packed_dim,
                tensor.meta.shape(),
                scheme.num_quants(),
                dtype_to_storage_type(DType::U32),
            );
        }
        // e2m1 is special because it has a native packed representation, `e2m1x2`.
        // It's internally stored as `u8` with a packing factor of 2.
        QuantStore::PackedNative(packed_dim) if scheme.value == QuantValue::E2M1 => {
            cubecl::std::tensor::into_contiguous_packed_ref(
                &client,
                values.binding(),
                out_values.binding(),
                packed_dim,
                tensor.meta.shape(),
                scheme.num_quants(),
                dtype_to_storage_type(DType::U8),
            );
        }
        _ => {
            cubecl::std::tensor::copy_into(
                &client,
                values.binding(),
                out_values.binding(),
                dtype_to_storage_type(dtype_value),
            );
        }
    }

    cubecl::std::tensor::copy_into(
        &client,
        scales.binding(),
        out_scales.binding(),
        dtype_to_storage_type(dtype_scales),
    );

    if let (Some(global), Some(out_global)) = (tensor.global(), output.global()) {
        let dtype_global = global.dtype;
        cubecl::std::tensor::copy_into(
            &client,
            global.binding(),
            out_global.binding(),
            dtype_to_storage_type(dtype_global),
        );
    }

    output
}

/// A tensor packed into storage tiles through cubek, and what burn does with one: the matmul it
/// was packed for reads it through its binding, every layout rewrite lays it back in rows first,
/// and a row kernel refuses it.
#[cfg(all(
    test,
    any(feature = "wgpu", feature = "cpu", feature = "cuda", feature = "hip")
))]
mod storage_tiled {
    use burn_backend::{DType, cubecl::dtype_to_storage_type};
    use burn_std::{Shape, TensorData};

    use crate::{
        CubeDevice,
        kernel::{
            into_contiguous,
            matmul::{MatmulStrategy, matmul},
            slice, untile,
        },
        ops::{from_data, into_data_sync, reshape},
        tensor::CubeTensor,
    };

    fn tensor(shape: &[usize], device: &CubeDevice, seed: u32) -> CubeTensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n as u32)
            .map(|i| {
                ((i.wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 97) as f32 / 97.0 - 0.5
            })
            .collect();
        from_data(TensorData::new(data, shape.to_vec()), device)
    }

    fn packed(tensor: &CubeTensor, tile: (usize, usize)) -> CubeTensor {
        let client = tensor.client.clone();
        let out = cubek::matmul::tiled::pack::pack(
            &client,
            tensor.clone().binding(),
            dtype_to_storage_type(tensor.dtype),
            tile,
        )
        .expect("the tile divides the matrix");
        CubeTensor::new(
            client,
            out.handle,
            *out.metadata,
            tensor.device.clone(),
            tensor.dtype,
        )
    }

    fn values(tensor: CubeTensor) -> Vec<f32> {
        into_data_sync(tensor).as_slice::<f32>().unwrap().to_vec()
    }

    fn assert_close(have: &[f32], want: &[f32], what: &str) {
        assert_eq!(have.len(), want.len(), "{what}: lengths differ");
        for (i, (h, w)) in have.iter().zip(want).enumerate() {
            assert!((h - w).abs() < 1e-3, "{what}: at {i}, got {h}, want {w}");
        }
    }

    /// The tile is one the matmul's plan can stage to; the selector honours it for this `m`.
    #[test]
    fn a_packed_weight_computes_the_same_product() {
        let device = CubeDevice::default();
        let (m, k, n) = (64, 256, 512);
        let lhs = tensor(&[m, k], &device, 1);
        let rhs = tensor(&[k, n], &device, 2);
        let weight = packed(&rhs, (32, 64));
        assert!(weight.meta.is_tiled());

        let plain = matmul(lhs.clone(), rhs, None, MatmulStrategy::Cube, DType::F32).unwrap();
        let tiled = matmul(lhs, weight, None, MatmulStrategy::Cube, DType::F32).unwrap();
        assert_eq!(tiled.meta.shape().as_slice(), &[m, n]);
        assert_close(&values(tiled), &values(plain), "packed weight");
    }

    #[test]
    fn untile_lays_the_rows_back() {
        let device = CubeDevice::default();
        let rhs = tensor(&[2, 64, 96], &device, 3);
        let weight = packed(&rhs, (16, 32));
        let back = untile(weight);
        assert!(!back.meta.is_tiled());
        assert_eq!(back.meta.shape().as_slice(), &[2, 64, 96]);
        assert_eq!(values(back), values(rhs));
    }

    /// The layout rewrites read rows: on a packed tensor they lay it back first, and agree
    /// with the same rewrite of the plain one.
    #[test]
    fn layout_rewrites_untile_first() {
        let device = CubeDevice::default();
        let rhs = tensor(&[64, 96], &device, 4);
        let weight = packed(&rhs, (16, 32));

        let reshaped = reshape(weight.clone(), Shape::new([32, 192]));
        assert!(!reshaped.meta.is_tiled());
        assert_eq!(
            values(reshaped),
            values(reshape(rhs.clone(), Shape::new([32, 192])))
        );

        let ranges = [8..40, 32..80];
        let sliced = slice(weight.clone(), &ranges);
        assert!(!sliced.meta.is_tiled());
        assert_eq!(values(sliced), values(slice(rhs.clone(), &ranges)));

        let contiguous = into_contiguous(weight);
        assert!(!contiguous.meta.is_tiled());
        assert_eq!(values(contiguous), values(rhs));
    }

    #[test]
    #[should_panic(expected = "storage-tiled")]
    fn a_row_kernel_refuses_a_packed_tensor() {
        let device = CubeDevice::default();
        let rhs = tensor(&[64, 96], &device, 5);
        let weight = packed(&rhs, (16, 32));
        let _ = weight.into_tensor_arg();
    }
}
