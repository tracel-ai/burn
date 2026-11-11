//! A tensor as cubek's tile-DSL routines read it: its values bound as they are, or, quantized,
//! packed under its scales.

use super::CubeTensor;
use burn_backend::DType;
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::tile::{Axis, launch::Refusal};
use cubek::matmul::kernels::tensor::{PackedTensor, PlainTensor, StoredTensor};

impl CubeTensor {
    /// This tensor bound as its values, at the element they are stored in.
    pub fn plain(self) -> PlainTensor {
        let elem = dtype_to_storage_type(self.dtype);
        PlainTensor::new(self.stored_binding(), elem)
    }

    /// This tensor, one matrix over `axes` in the order its shape states them, as it is stored:
    /// plain, or, quantized, packed under its scales ([`PackedTensor::of_quantized`]).
    ///
    /// # Errors
    ///
    /// For a quantized tensor, what [`PackedTensor::of_quantized`] declines.
    pub fn stored(self, axes: [Axis; 2]) -> Result<StoredTensor, Refusal> {
        let DType::QFloat(scheme) = self.dtype else {
            return Ok(StoredTensor::Plain(self.plain()));
        };
        let block_scales = self
            .scales()
            .expect("a quantized tensor holds its scales")
            .plain();
        let global = self.global().map(CubeTensor::plain);
        let client = self.client.clone();
        PackedTensor::of_quantized(
            &client,
            self.stored_binding(),
            scheme,
            block_scales,
            global,
            axes,
        )
        .map(StoredTensor::Packed)
    }
}

// Quantizes on a device: the Linux runners have no GPU for the test runtime's wgpu.
#[cfg(all(test, any(feature = "cuda", feature = "vulkan", feature = "metal")))]
mod tests {
    use burn_backend::{
        DType, TensorData,
        ops::QTensorOps,
        quantization::{QuantScheme, QuantValue, ScaleDtype},
    };
    use cubecl::quant::scheme::QuantStore;
    use cubek::matmul::kernels::tensor::StoredTensor;
    use cubek::std::matmul_axes::{K, N};

    use crate::{
        CubeBackend, CubeDevice,
        ops::{from_data, swap_dims},
        tensor::CubeTensor,
    };

    /// A tied embedding read as the vocabulary head: `[n, k]` packed along `k`, transposed to
    /// `[k, n]`, its words now running along the view's first axis. The matmul reads it as stored,
    /// as it reads the projections; refused, it would dequantize the whole head on every call.
    #[test]
    fn a_transposed_packed_matrix_is_read_as_stored() {
        let device = CubeDevice::default();
        let (n, k) = (96, 64);
        let values: Vec<f32> = (0..n * k).map(|i| (i % 29) as f32 / 29. - 0.5).collect();
        let embedding: CubeTensor = from_data(TensorData::new(values, [n, k]), &device);
        let scheme = QuantScheme::default()
            .with_value(QuantValue::E2M1)
            .per_block([16], ScaleDtype::UE4M3)
            .per_tensor(ScaleDtype::F32)
            .with_store(QuantStore::PackedU32(0));
        let head = swap_dims(CubeBackend::quantize_dynamic(embedding, &scheme), 0, 1);

        let stored = head.stored([K, N]);
        assert!(
            matches!(stored, Ok(StoredTensor::Packed(_))),
            "a transposed packed matrix is read packed, got {stored:?}"
        );
    }

    /// An NVFP4 projection a load laid in tile order for a device's instruction is read packed as
    /// it is stored, and multiplies as the same weight in rows does. Refused, the matmul fell back
    /// to dequantizing it, which a storage-tiled tensor refuses: the product panicked. Runs where
    /// the device stores a tile for the weight against a bf16 activation.
    #[test]
    fn a_packed_matrix_in_tile_order_is_read_as_stored() {
        use burn_backend::{TensorPrimitive, ops::FloatTensorOps};
        use burn_std::FloatDType;
        use cubecl::ir::{ElemType, FloatKind};
        use cubek::matmul::tiled::storage::{StoredValues, stored_tile};
        use cubek::microkernel::matmul::MatmulOperand;
        use cubek::std::Hardware;

        let device = CubeDevice::default();
        // Whole tiles of every tile a device stores NVFP4 in: 128 × 8 on a Blackwell.
        let (m, n, k) = (24, 64, 256);
        let values: Vec<f32> = (0..n * k).map(|i| (i % 29) as f32 / 29. - 0.5).collect();
        let scheme = QuantScheme::default()
            .with_value(QuantValue::E2M1)
            .per_block([16], ScaleDtype::UE4M3)
            .per_tensor(ScaleDtype::F32)
            .with_store(QuantStore::PackedU32(0));
        // As a checkpoint holds a projection, `[n, k]` packed along `k`, served as `[k, n]`.
        let weight = from_data(TensorData::new(values, [n, k]), &device);
        let rows = swap_dims(CubeBackend::quantize_dynamic(weight, &scheme), 0, 1);
        let hw = Hardware::from_client(&rows.client);
        let DType::QFloat(served) = rows.dtype else {
            unreachable!("a quantized weight")
        };
        let activation = ElemType::Float(FloatKind::BF16);
        let Some(tile) = stored_tile(
            &hw,
            MatmulOperand::Rhs,
            StoredValues::Packed(served),
            activation,
        ) else {
            return;
        };
        let [tile_k, tile_n] = MatmulOperand::Rhs.axes().map(|axis| tile.extent(axis));
        let tiled = tile_order::laid(rows.clone(), (tile_k, tile_n));

        let stored = tiled.clone().stored([K, N]);
        assert!(
            matches!(stored, Ok(StoredTensor::Packed(_))),
            "a packed matrix in tile order is read packed, got {stored:?}"
        );

        let input: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 / 13. - 0.5).collect();
        let input = CubeBackend::float_cast(
            from_data(TensorData::new(input, [m, k]), &device),
            FloatDType::BF16,
        );
        let product = |weight: CubeTensor| {
            let out = CubeBackend::q_matmul(
                TensorPrimitive::Float(input.clone()),
                TensorPrimitive::QFloat(weight),
            );
            let TensorPrimitive::Float(out) = out else {
                unreachable!("a product against a float input is float")
            };
            burn_std::future::block_on(CubeBackend::float_into_data(out))
                .expect("the product reads back")
                .convert::<f32>()
        };
        product(tiled).assert_approx_eq::<f32>(
            &product(rows),
            burn_std::Tolerance::relative(1e-2).set_absolute(1e-2),
        );
    }

    /// The tile-order writer a load runs for a kernel that reads tile order, as a test lays a
    /// weight out: Burn writes no tile order of its own.
    mod tile_order {
        use burn_backend::cubecl::dtype_to_storage_type;
        use burn_backend::{DType, quantization::QParamTensor};
        use burn_std::{Metadata, Shape, quantization::QParams};
        use cubecl::server::Handle;
        use cubecl::tile::kind::Field;
        use cubek::matmul::tiled::storage;
        use cubek::std::Hardware;
        use cubek::std::layout::tile_order::{self, TileOrder};
        use cubek::std::matmul_axes::N;

        use crate::{kernel::into_contiguous, tensor::CubeTensor};

        /// `weight`, a `[k, n]` matrix packed down `k` of an `[n, k]` buffer, in tile order in
        /// `tile`, `(k, n)`: its values relaid out in the order the device's readers take, its
        /// scales in tiles of their own where that order keeps them so, gathered with its
        /// per-tensor scale into one allocation, its metadata stating the values over `[k, n]`.
        pub fn laid(weight: CubeTensor, tile: (usize, usize)) -> CubeTensor {
            let DType::QFloat(scheme) = weight.dtype else {
                panic!("tile order stores a packed weight")
            };
            let hw = Hardware::from_client(&weight.client);
            let [k, n] = [weight.meta.shape()[0], weight.meta.shape()[1]];
            let order = storage::tile_order_on(&hw, &scheme, tile, (k, n))
                .expect("tile order stores the weight");
            let word = dtype_to_storage_type(DType::U32);
            let words_per_load = hw.load_values(word);
            let per_word = scheme.num_quants();
            let row_step = weight.meta.strides()[1];
            let (client, device) = (weight.client.clone(), weight.device.clone());
            let relayout = |source: CubeTensor, axes: &[_], field, storage| {
                storage::tile(
                    &client,
                    source.stored_binding(),
                    axes,
                    word,
                    Some(field),
                    storage,
                )
                .expect("the weight closes whole tiles")
            };

            let blocks = k / order.block();
            let values = CubeTensor::new(
                client.clone(),
                weight.handle.clone(),
                Metadata::new(
                    Shape::new([n, blocks, order.block()]),
                    vec![row_step * per_word, order.block(), 1],
                ),
                device.clone(),
                DType::U32,
            );
            let values = relayout(
                values,
                &tile_order::values_axes(N),
                Field::from(scheme.value),
                order.values_storage(N, words_per_load),
            );
            let scales = weight.scales().expect("a packed weight has block scales");
            let [scale_rows, scale_columns] = [scales.meta.shape()[0], scales.meta.shape()[1]];
            let [row_step, column_step] = [scales.meta.strides()[0], scales.meta.strides()[1]];
            let scales = into_contiguous(CubeTensor {
                meta: Box::new(Metadata::new(
                    [scale_columns, scale_rows],
                    [column_step, row_step],
                )),
                ..scales
            });
            let rows_a_read = order.rows_a_load(N, words_per_load);
            let word_bytes = DType::U32.size();
            let (scales, scales_metadata, scales_bytes) = match order.scales_storage(N, rows_a_read)
            {
                Some(storage) => {
                    let block_scale = scheme.block_scale().expect("tile order keeps block scales");
                    let tiled = relayout(
                        CubeTensor::new(
                            client.clone(),
                            scales.handle,
                            Metadata::new(Shape::new([n, blocks]), vec![blocks, 1]),
                            device.clone(),
                            DType::U32,
                        ),
                        &tile_order::scales_axes(N),
                        Field::of_scale(block_scale.dtype),
                        storage,
                    );
                    let bytes =
                        words_spanned(&tiled.metadata, TileOrder::SCALES_PER_WORD) * word_bytes;
                    (
                        tiled.handle,
                        tile_order::matrix_scales(&tiled.metadata),
                        bytes,
                    )
                }
                None => (
                    scales.handle,
                    Metadata::new(Shape::new([blocks, n]), vec![1, blocks]),
                    blocks * n,
                ),
            };
            let global = weight.global();

            let align = |bytes: usize| {
                bytes.next_multiple_of(client.properties().memory.alignment as usize)
            };
            let values_bytes = words_spanned(&values.metadata, per_word) * word_bytes;
            let scales_at = align(values_bytes);
            let global_at = align(scales_at + scales_bytes);
            let total = match global {
                Some(_) => global_at + DType::F32.size(),
                None => scales_at + scales_bytes,
            };
            let buffer = client.empty(total);
            let place = |from: Handle, at: usize, bytes: usize| {
                let words = |handle| {
                    let shape = Shape::new([bytes / word_bytes]);
                    CubeTensor::new_contiguous(
                        client.clone(),
                        device.clone(),
                        shape,
                        handle,
                        DType::U32,
                    )
                    .stored_binding()
                };
                let mut to = buffer.clone();
                to.offset_start = Some(at as u64);
                to.offset_end = Some((total - at - bytes) as u64);
                cubecl::std::tensor::copy_into(&client, words(from), words(to), word);
            };
            place(values.handle, 0, values_bytes);
            place(scales, scales_at, scales_bytes);
            if let Some(global) = &global {
                place(global.handle.clone(), global_at, DType::F32.size());
            }

            let mut stored = CubeTensor::new(
                client,
                buffer,
                tile_order::matrix_values(&values.metadata),
                device,
                weight.dtype,
            );
            stored.qparams = Some(QParams {
                scales: QParamTensor {
                    offset_start: scales_at,
                    offset_end: total - scales_at - scales_bytes,
                    metadata: scales_metadata,
                    dtype: DType::U8,
                },
                global: global.map(|_| QParamTensor {
                    offset_start: global_at,
                    offset_end: 0,
                    metadata: Metadata::new(Shape::new([1]), vec![1]),
                    dtype: DType::F32,
                }),
            });
            stored
        }

        /// The `u32` words a buffer stated as `metadata` spans, `per_word` of its elements to a
        /// word, the padding between its strips included.
        fn words_spanned(metadata: &Metadata, per_word: usize) -> usize {
            let dims: Vec<(usize, usize)> = metadata
                .shape()
                .iter()
                .zip(metadata.strides().iter())
                .map(|(&extent, &stride)| (extent, stride))
                .collect();
            match cubecl::tile::Geometry::new(&dims).span() {
                0 => 0,
                elements => (elements - 1) / per_word + 1,
            }
        }
    }
}
