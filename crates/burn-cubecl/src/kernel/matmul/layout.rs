use crate::{CubeBackend, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, get_or_init_device_settings};
use cubek::matmul::tiled::storage::{StorageLevels, StoredValues, stored_tile, tile};
use cubek::microkernel::matmul::MatmulOperand;
use cubek::std::{
    Hardware,
    matmul_axes::{K, N},
};

/// `weight`, `[.., k, n]`, copied into the storage tiles this device's matmul reads its second
/// input from ([`stored_tile`]) where a tile divides `k` and `n`, for a product contracted at the
/// device's float dtype; as it lies otherwise.
///
/// A tile holds its rows one after another, each split at the widest vector read that divides it,
/// and the tiles lie row-major. A quantized weight, one already storage-tiled, and one of fewer
/// than two dims or with an empty dim come back as they lie.
pub fn lay_out_for_matmul(weight: CubeTensor) -> CubeTensor {
    let [.., k, n] = weight.meta.shape()[..] else {
        return weight;
    };
    if weight.qparams.is_some() || weight.meta.is_tiled() || weight.meta.num_elements() == 0 {
        return weight;
    }
    let (client, device, dtype) = (weight.client.clone(), weight.device.clone(), weight.dtype);
    let element = dtype_to_storage_type(dtype);
    let contracted = DType::from(get_or_init_device_settings::<CubeBackend>(&device).float_dtype);
    let Some(stored) = stored_tile(
        &Hardware::from_client(&client),
        MatmulOperand::Rhs,
        StoredValues::Elements(element),
        dtype_to_storage_type(contracted),
    ) else {
        return weight;
    };
    let (rows, columns) = (stored.extent(K), stored.extent(N));
    if !k.is_multiple_of(rows) || !n.is_multiple_of(columns) {
        return weight;
    }

    // A reader takes a stated piece whole, so the finest piece is the read the tile is read in.
    let read = client
        .io_optimized_vector_sizes(element.size())
        .filter(|&width| columns.is_multiple_of(width))
        .max()
        .unwrap_or(1);
    // A piece of one is no piece: stated, it would be a dim of the buffer.
    let pieces: Vec<_> = [(N, read), (N, columns / read), (K, rows)]
        .into_iter()
        .filter(|&(_, count)| count > 1)
        .collect();
    let storage = StorageLevels::new(&pieces).grid(&[N, K]);

    let output = tile(&client, weight.binding(), &[K, N], element, None, storage)
        .expect("a tile that divides the matrix closes it in whole tiles");
    CubeTensor::new(client, output.handle, *output.metadata, device, dtype)
}
