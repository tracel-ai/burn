//! A sum or mean with few outputs over a long axis, such as the marginals of a mutual information
//! over `V` pixels (`[1, 32, V]` over `V`) or the gradient of a per-image minimum (`[1, 1, V]` over
//! `V`).
//!
//! The unit routine gives each output one thread walking its whole axis, so a handful of outputs
//! leave the device idle. Here the axis is cut into chunks: one thread sums each chunk of each
//! output, then one thread per output adds its chunks in order.

use burn_backend::DType;
use burn_backend::cubecl::{dtype_to_elem_type, dtype_to_storage_type};
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use cubek::reduce::{ReduceDtypes, components::instructions::ReduceOperationConfig};

use crate::{kernel::utils::address_type, ops::numeric::empty_device_dtype, tensor::CubeTensor};

/// The most outputs this route takes: past them the unit routine has threads enough.
const MAX_OUTPUTS: usize = 256;
/// The shortest axis this route takes.
const MIN_LENGTH: usize = 1 << 16;
/// The length of the axis one thread sums before the chunks are added together.
const CHUNK: usize = 4096;

/// Each thread sums one chunk of one output's axis into `partial`, laid out `[outputs, chunks]`.
#[cube(launch, address_type = "dynamic")]
fn sum_chunk_kernel<F: Float>(
    input: &Tensor<F>,
    partial: &mut Tensor<F>,
    output: &Tensor<F>,
    axis: usize,
    chunk: usize,
    chunks: usize,
    #[define(F)] _dtype: ElemType,
) {
    if ABSOLUTE_POS >= partial.len() {
        terminate!();
    }
    let reduction = ABSOLUTE_POS / chunks;
    let c = ABSOLUTE_POS % chunks;
    let mut at = 0;
    for dim in 0..input.rank() {
        let coordinate = (reduction / output.stride(dim)) % output.shape(dim);
        at += coordinate * input.stride(dim);
    }
    let step = input.stride(axis);
    let start = c * chunk;
    at += start * step;
    let rest = input.shape(axis) - start;
    let length = if rest < chunk { rest } else { chunk };
    let mut acc = F::new(0.0);
    for _ in 0..length / 4 {
        let v0 = input[at];
        let v1 = input[at + step];
        let v2 = input[at + 2 * step];
        let v3 = input[at + 3 * step];
        acc += v0;
        acc += v1;
        acc += v2;
        acc += v3;
        at += 4 * step;
    }
    for _ in 0..length % 4 {
        acc += input[at];
        at += step;
    }
    partial[ABSOLUTE_POS] = acc;
}

/// Each thread adds one output's chunks in order, divided by the axis length for a mean.
#[cube(launch, address_type = "dynamic")]
fn sum_chunks_kernel<F: Float>(
    partial: &Tensor<F>,
    output: &mut Tensor<F>,
    chunks: usize,
    length: usize,
    #[comptime] mean: bool,
    #[define(F)] _dtype: ElemType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let mut acc = F::new(0.0);
    for c in 0..chunks {
        acc += partial[ABSOLUTE_POS * chunks + c];
    }
    if comptime!(mean) {
        output[ABSOLUTE_POS] = acc / F::cast_from(length);
    } else {
        output[ABSOLUTE_POS] = acc;
    }
}

/// Whether the chunked route takes this reduction in place of the unit routine: a float sum or
/// mean accumulated in its own type, over an axis of at least [`MIN_LENGTH`], into at most
/// [`MAX_OUTPUTS`] values.
pub(crate) fn applies(
    input: &CubeTensor,
    dim: usize,
    config: ReduceOperationConfig,
    dtypes: &ReduceDtypes,
) -> bool {
    let shape = input.meta.shape();
    let float = dtype_to_elem_type(input.dtype);
    matches!(
        config,
        ReduceOperationConfig::Sum | ReduceOperationConfig::Mean
    ) && matches!(input.dtype, DType::F32 | DType::F64)
        && dtypes.input == float
        && dtypes.accumulation == float
        && dtypes.output == float
        && input.qparams.is_none()
        && shape[dim] >= MIN_LENGTH
        && shape.num_elements() / shape[dim] <= MAX_OUTPUTS
}

/// `output` = the sum (or mean) of `input` over `dim` by the chunked route, for a reduction it
/// [`applies`] to. `output` is the contiguous tensor the reduction writes.
pub(crate) fn reduce_long_axis(input: CubeTensor, output: CubeTensor, dim: usize, mean: bool) {
    let client = input.client.clone();
    let dtype = input.dtype;
    let length = input.meta.shape()[dim];
    let chunks = length.div_ceil(CHUNK);
    let outputs = output.meta.shape().num_elements();
    let partial = empty_device_dtype(
        client.clone(),
        input.device.clone(),
        [outputs * chunks].into(),
        dtype,
    );
    let cube_dim = CubeDim::new_1d(256);
    sum_chunk_kernel::launch(
        &client,
        calculate_cube_count_elemwise(&client, outputs * chunks, cube_dim),
        cube_dim,
        address_type!(input, partial, output),
        input.into_tensor_arg(),
        partial.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
        dim,
        CHUNK,
        chunks,
        dtype_to_storage_type(dtype),
    );
    let cube_dim = CubeDim::new_1d(32);
    sum_chunks_kernel::launch(
        &client,
        calculate_cube_count_elemwise(&client, outputs, cube_dim),
        cube_dim,
        address_type!(partial, output),
        partial.into_tensor_arg(),
        output.into_tensor_arg(),
        chunks,
        length,
        mean,
        dtype_to_storage_type(dtype),
    );
}
