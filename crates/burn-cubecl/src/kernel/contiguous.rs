use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, TensorMetadata};
use cubecl::quant::scheme::{QuantStore, QuantValue};
use cubecl::server::MemoryLayoutStrategy;

use crate::{ops::empty_qtensor, tensor::CubeTensor};

/// A storage-tiled tensor laid out in rows again, through cubek's untile; a plain tensor as it
/// is. cubek's matmul reads the tiles and the elementwise kernels walk them in place; every other
/// kernel reads rows, through [`CubeTensor::binding`], which lays them back here.
///
/// # Panics
///
/// A quantized storage-tiled tensor: it is read only by the kernel it was tiled for, and
/// nothing lays its packed values and their scales back as rows.
pub fn untile(tensor: CubeTensor) -> CubeTensor {
    if !tensor.meta.is_tiled() {
        return tensor;
    }
    assert!(
        tensor.qparams.is_none(),
        "untile: a storage-tiled quantized tensor is read only by the kernel it was tiled for; \
         nothing lays its packed values back as rows"
    );
    let (client, device, dtype) = (tensor.client.clone(), tensor.device.clone(), tensor.dtype);
    let output = cubek::matmul::tiled::storage::untile(
        &client,
        tensor.stored_binding(),
        dtype_to_storage_type(dtype),
    )
    .expect("a storage-tiled binding describes its own tiles");
    CubeTensor::new(client, output.handle, *output.metadata, device, dtype)
}

/// Make a jit tensor contiguous.
pub fn into_contiguous(tensor: CubeTensor) -> CubeTensor {
    // A tiled buffer's strides order its tiles, not a matrix's rows: contiguous says nothing.
    let tensor = untile(tensor);
    if tensor.is_contiguous() {
        return tensor;
    }
    copy_contiguous(tensor)
}

/// Copy a jit tensor into a new buffer holding exactly its elements in row-major order, even when
/// it is contiguous already.
pub fn copy_contiguous(tensor: CubeTensor) -> CubeTensor {
    let tensor = untile(tensor);
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

/// A tensor stored in storage tiles through cubek, and what burn does with one: the matmul reads
/// its tiles, the dim rewrites and the elementwise ops keep them, and every other kernel reads it
/// as rows.
#[cfg(all(
    test,
    any(
        feature = "wgpu",
        feature = "metal",
        feature = "cpu",
        feature = "cuda",
        feature = "hip"
    )
))]
mod storage_tiled {
    use burn_backend::{
        DType, FloatDType, Scalar, TensorMetadata, cubecl::dtype_to_storage_type,
        ops::FloatTensorOps,
    };
    use burn_std::{Shape, TensorData};

    use crate::{
        CubeBackend, CubeDevice,
        kernel::{
            into_contiguous,
            matmul::{MatmulStrategy, matmul},
            slice, untile,
        },
        ops::{expand, from_data, into_data_sync, permute, reshape, swap_dims},
        tensor::CubeTensor,
    };

    /// The tile is one the matmul's plan can stage to; the selector honours it for this `m`.
    #[test]
    fn a_tiled_weight_computes_the_same_product() {
        let device = CubeDevice::default();
        let (m, k, n) = (64, 256, 512);
        let lhs = tensor(&[m, k], &device, 1);
        let rhs = tensor(&[k, n], &device, 2);
        let weight = tiled(&rhs, (32, 64));
        assert!(weight.meta.is_tiled());

        let plain = matmul(lhs.clone(), rhs, None, MatmulStrategy::Cube, DType::F32).unwrap();
        let from_tiles = matmul(lhs, weight, None, MatmulStrategy::Cube, DType::F32).unwrap();
        assert_eq!(from_tiles.meta.shape().as_slice(), &[m, n]);
        assert_close(&values(from_tiles), &values(plain), "tiled weight");
    }

    /// A decode step is one row against the weight: the routines that serve it read the tiles
    /// too, whether cubek's rules pick the routine or autotune races them.
    #[test]
    fn a_tiled_weight_computes_the_same_decode_product() {
        let device = CubeDevice::default();
        let lhs = tensor(&[1, 256], &device, 1);
        let rhs = tensor(&[256, 512], &device, 2);
        let weight = tiled(&rhs, (32, 64));
        let plain = matmul(lhs.clone(), rhs, None, MatmulStrategy::Cube, DType::F32).unwrap();
        let plain = values(plain);
        for strategy in strategies() {
            let from_tiles = matmul(lhs.clone(), weight.clone(), None, strategy, DType::F32);
            assert_close(&values(from_tiles.unwrap()), &plain, "decode, tiled weight");
        }
    }

    /// Training multiplies by the transposed weight: a transpose keeps the tiles, stored a
    /// column at a time now, and the product is the same whether a routine reads them or every
    /// routine declines them and the product reads rows.
    #[test]
    fn a_transposed_tiled_weight_computes_the_same_product() {
        let device = CubeDevice::default();
        let lhs = tensor(&[64, 512], &device, 1);
        let rhs = tensor(&[256, 512], &device, 2);
        let weight = swap_dims(tiled(&rhs, (32, 64)), 0, 1);
        assert!(weight.meta.is_tiled());
        let plain = swap_dims(rhs, 0, 1);
        let plain = matmul(lhs.clone(), plain, None, MatmulStrategy::Cube, DType::F32).unwrap();
        let plain = values(plain);
        for strategy in strategies() {
            let from_tiles = matmul(lhs.clone(), weight.clone(), None, strategy, DType::F32);
            assert_close(
                &values(from_tiles.unwrap()),
                &plain,
                "transposed tiled weight",
            );
        }
    }

    #[test]
    fn untile_lays_the_rows_back() {
        let device = CubeDevice::default();
        let rhs = tensor(&[2, 64, 96], &device, 3);
        let weight = tiled(&rhs, (16, 32));
        let back = untile(weight);
        assert!(!back.meta.is_tiled());
        assert_eq!(back.meta.shape().as_slice(), &[2, 64, 96]);
        assert_eq!(values(back), values(rhs));
    }

    #[test]
    fn multi_axis_sum_uses_the_logical_shape_of_tiled_storage() {
        let device = CubeDevice::default();
        let input = tensor(&[2, 64, 96], &device, 3);
        let weight = tiled(&input, (16, 32));
        assert!(weight.meta.is_tiled());

        let expected = CubeBackend::float_sum_dims(input, &[0, 2]);
        let actual = CubeBackend::float_sum_dims(weight, &[0, 2]);
        assert_eq!(actual.shape(), expected.shape());
        assert_close(&values(actual), &values(expected), "tiled reduction");
    }

    /// Swapping, permuting, broadcasting and adding or dropping dims of one rewrite metadata
    /// alone: the tiles stay where they are, and every value reads where the same rewrite of
    /// the rows puts it.
    #[test]
    fn dim_rewrites_keep_the_tiles() {
        let device = CubeDevice::default();
        let rhs = tensor(&[64, 96], &device, 4);
        let weight = tiled(&rhs, (16, 32));
        let kept = |rewritten: CubeTensor, of_rows: CubeTensor, what: &str| {
            assert!(rewritten.meta.is_tiled(), "{what} keeps the tiles");
            assert_eq!(rewritten.shape(), of_rows.shape(), "{what}");
            assert_eq!(
                values(rewritten),
                values(into_contiguous(of_rows)),
                "{what}"
            );
        };

        kept(
            swap_dims(weight.clone(), 0, 1),
            swap_dims(rhs.clone(), 0, 1),
            "swap",
        );
        kept(
            permute(weight.clone(), &[1, 0]),
            permute(rhs.clone(), &[1, 0]),
            "permute",
        );
        let batched = Shape::new([3, 64, 96]);
        kept(
            expand(weight.clone(), batched.clone()),
            expand(rhs.clone(), batched),
            "expand",
        );
        let unit = Shape::new([1, 64, 1, 96]);
        let unsqueezed = reshape(weight.clone(), unit.clone());
        kept(unsqueezed.clone(), reshape(rhs.clone(), unit), "unsqueeze");
        kept(reshape(unsqueezed, Shape::new([64, 96])), rhs, "squeeze");
    }

    /// A reshape that regroups the tiled dims, and a slice, read the tensor as rows.
    #[test]
    fn other_layout_rewrites_read_rows() {
        let device = CubeDevice::default();
        let rhs = tensor(&[64, 96], &device, 4);
        let weight = tiled(&rhs, (16, 32));

        let reshaped = reshape(weight.clone(), Shape::new([32, 192]));
        assert!(!reshaped.meta.is_tiled());
        assert_eq!(
            values(reshaped),
            values(reshape(rhs.clone(), Shape::new([32, 192])))
        );

        let ranges = [8..40, 32..80];
        let sliced = slice(weight, &ranges);
        assert!(!sliced.meta.is_tiled());
        assert_eq!(values(sliced), values(slice(rhs, &ranges)));
    }

    /// An elementwise op walks a tiled operand's tiles in place, every other operand restated
    /// over them, and its output is tiled the same way: no relayout on the way in or out.
    #[test]
    fn elementwise_ops_keep_the_tiles() {
        let device = CubeDevice::default();
        let rhs = tensor(&[64, 96], &device, 5);
        let weight = tiled(&rhs, (16, 32));
        let rows = tensor(&[64, 96], &device, 6);
        let bias = tensor(&[1, 96], &device, 7);
        let other_tiles = tiled(&rows, (32, 16));
        let kept = |out: CubeTensor, want: CubeTensor, what: &str| {
            assert_eq!(
                out.meta.tiling, weight.meta.tiling,
                "{what} keeps the tiles"
            );
            assert_close(&values(out), &values(want), what);
        };

        let one = Scalar::from(1.0f32);
        kept(
            CubeBackend::float_add_scalar(weight.clone(), one),
            CubeBackend::float_add_scalar(rhs.clone(), one),
            "w + 1",
        );
        kept(
            CubeBackend::float_add(weight.clone(), bias.clone()),
            CubeBackend::float_add(rhs.clone(), bias),
            "w + bias",
        );
        kept(
            CubeBackend::float_sub(rows.clone(), weight.clone()),
            CubeBackend::float_sub(rows.clone(), rhs.clone()),
            "x - w",
        );
        kept(
            CubeBackend::float_mul(weight.clone(), weight.clone()),
            CubeBackend::float_mul(rhs.clone(), rhs.clone()),
            "w * w",
        );
        kept(
            CubeBackend::float_add(weight.clone(), other_tiles),
            CubeBackend::float_add(rhs.clone(), rows),
            "w + other tiles",
        );
        kept(
            CubeBackend::float_exp(weight.clone()),
            CubeBackend::float_exp(rhs.clone()),
            "exp(w)",
        );
        kept(
            CubeBackend::float_cast(weight.clone(), FloatDType::F16),
            CubeBackend::float_cast(rhs, FloatDType::F16),
            "cast",
        );
    }

    /// A kernel with no notion of tiles reads a tiled tensor as rows, through a copy, instead of
    /// refusing it.
    #[test]
    fn a_row_kernel_reads_a_tiled_tensor_as_rows() {
        let device = CubeDevice::default();
        let rhs = tensor(&[64, 96], &device, 8);
        let weight = tiled(&rhs, (16, 32));

        assert_close(
            &values(CubeBackend::float_sum_dim(weight.clone(), 1)),
            &values(CubeBackend::float_sum_dim(rhs.clone(), 1)),
            "sum along a row",
        );
        assert_close(
            &values(CubeBackend::float_sum(weight)),
            &values(CubeBackend::float_sum(rhs)),
            "sum",
        );
    }

    /// The same tensors through fusion: every fused kernel reads them laid back in rows.
    #[cfg(feature = "fusion")]
    mod fused {
        use super::{assert_close, tensor, tiled, values};
        use crate::{
            CubeBackend, CubeDevice,
            fusion::{CubeFusionHandle, FusionCubeRuntime},
            ops::into_data_sync,
            tensor::CubeTensor,
        };
        use burn_backend::{Scalar, ops::FloatTensorOps};
        use burn_fusion::{
            ExecutionError, Fusion, FusionTensor,
            custom::{
                CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, StreamId,
                TensorIr,
            },
        };

        type Fused = Fusion<CubeBackend>;

        /// `w * 2 + 1`, then `exp`, fuse into one elementwise block.
        #[test]
        fn a_fused_elementwise_chain_reads_the_tiles_as_rows() {
            let device = CubeDevice::default();
            let rows = tensor(&[64, 96], &device, 9);
            let chain = |w| {
                let w = Fused::float_mul_scalar(w, Scalar::from(2.0f32));
                let w = Fused::float_add_scalar(w, Scalar::from(1.0f32));
                Fused::float_exp(w)
            };

            assert_close(
                &values(resolve(chain(stored_in_tiles(fusion_tensor(
                    &rows, &device,
                ))))),
                &values(resolve(chain(fusion_tensor(&rows, &device)))),
                "exp(2w + 1)",
            );
        }

        /// A fused reduce.
        #[test]
        fn a_fused_reduce_reads_the_tiles_as_rows() {
            let device = CubeDevice::default();
            let rows = tensor(&[64, 96], &device, 12);
            let reduce =
                |w| Fused::float_sum_dim(Fused::float_mul_scalar(w, Scalar::from(2.0f32)), 1);

            assert_close(
                &values(resolve(reduce(stored_in_tiles(fusion_tensor(
                    &rows, &device,
                ))))),
                &values(resolve(reduce(fusion_tensor(&rows, &device)))),
                "sum of 2w along a row",
            );
        }

        /// A transpose fused into an elementwise block: a view, whose strides are rewritten on the
        /// rows.
        #[test]
        fn a_fused_view_reads_the_tiles_as_rows() {
            let device = CubeDevice::default();
            let rows = tensor(&[64, 96], &device, 13);
            let view =
                |w| Fused::float_mul_scalar(Fused::float_swap_dims(w, 0, 1), Scalar::from(3.0f32));

            assert_close(
                &values(resolve(view(stored_in_tiles(fusion_tensor(
                    &rows, &device,
                ))))),
                &values(resolve(view(fusion_tensor(&rows, &device)))),
                "3 wᵀ",
            );
        }

        /// Stores a tensor in `(16, 32)` tiles inside the fusion stream, as a backend extension
        /// that lays a weight out does.
        #[derive(Debug)]
        struct StoreInTiles {
            desc: CustomOpIr,
        }

        impl Operation<FusionCubeRuntime> for StoreInTiles {
            fn execute(
                &self,
                handles: &mut HandleContainer<CubeFusionHandle>,
            ) -> Result<(), ExecutionError> {
                let ([input], [output]) = self.desc.as_fixed();
                let rows = handles.get_float_tensor::<CubeBackend>(input);
                handles.register_float_tensor::<CubeBackend>(&output.id, tiled(&rows, (16, 32)));
                Ok(())
            }
        }

        fn stored_in_tiles(
            tensor: FusionTensor<FusionCubeRuntime>,
        ) -> FusionTensor<FusionCubeRuntime> {
            let client = tensor.client.clone();
            let input = tensor.into_ir();
            let output = TensorIr::uninit(
                client.create_empty_handle(),
                input.shape.clone(),
                input.dtype,
            );
            let desc = CustomOpIr::new("test::store_in_tiles", &[input], &[output]);
            client
                .register(
                    StreamId::current(),
                    OperationIr::Custom(desc.clone()),
                    StoreInTiles { desc },
                )
                .output()
        }

        fn fusion_tensor(
            tensor: &CubeTensor,
            device: &CubeDevice,
        ) -> FusionTensor<FusionCubeRuntime> {
            Fused::float_from_data(into_data_sync(tensor.clone()), device)
        }

        /// Runs the stream up to `tensor` and hands back the tensor it holds.
        fn resolve(tensor: FusionTensor<FusionCubeRuntime>) -> CubeTensor {
            tensor
                .client
                .clone()
                .resolve_tensor_float::<CubeBackend>(tensor)
                .unwrap()
        }
    }

    /// Every strategy a product may run under: cubek's rules, and autotune where it is built.
    fn strategies() -> Vec<MatmulStrategy> {
        vec![
            MatmulStrategy::Cube,
            #[cfg(feature = "autotune")]
            MatmulStrategy::Autotune,
        ]
    }

    fn tensor(shape: &[usize], device: &CubeDevice, seed: u32) -> CubeTensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n as u32)
            .map(|i| {
                ((i.wrapping_mul(2654435761).wrapping_add(seed) >> 8) % 97) as f32 / 97.0 - 0.5
            })
            .collect();
        from_data(TensorData::new(data, shape.to_vec()), device)
    }

    /// `tensor` stored in `(rows, cols)` tiles, a tile's rows one after another and the tiles
    /// row-major.
    fn tiled(tensor: &CubeTensor, (rows, cols): (usize, usize)) -> CubeTensor {
        use cubek::matmul::tiled::storage::{Axis, StorageLevels, tile};
        const ROW: Axis = Axis(0);
        const COL: Axis = Axis(1);
        let client = tensor.client.clone();
        let storage = StorageLevels::new(&[(COL, cols), (ROW, rows)]).grid(&[COL, ROW]);
        let out = tile(
            &client,
            tensor.clone().binding(),
            &[ROW, COL],
            dtype_to_storage_type(tensor.dtype),
            None,
            storage,
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

    /// The values in logical order, read at `f32` whatever the tensor holds.
    fn values(tensor: CubeTensor) -> Vec<f32> {
        into_data_sync(tensor)
            .convert::<f32>()
            .try_to_vec::<f32>()
            .unwrap()
    }

    fn assert_close(have: &[f32], want: &[f32], what: &str) {
        assert_eq!(have.len(), want.len(), "{what}: lengths differ");
        for (i, (h, w)) in have.iter().zip(want).enumerate() {
            assert!(
                (h - w).abs() < 1e-2 * w.abs().max(1.0),
                "{what}: at {i}, got {h}, want {w}"
            );
        }
    }
}
