use burn_tensor::{DType, Shape};
use cubecl::attention::{
    Strategy,
    components::{AttentionElems, AttentionSetupError},
};

use crate::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};

/// Launch a flash attention kernel
pub fn flash_attention<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    out_dtype: DType,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    let client = &query.client;
    let device = &query.device;

    let num_batches = query.shape.dims[0];
    let num_heads = query.shape.dims[1];
    let seq_q = query.shape.dims[2];
    let val_dim = value.shape.dims[3];
    let out_shape = Shape::new([num_batches, num_heads, seq_q, val_dim]);

    let out = empty_device_dtype::<R>(client.clone(), device.clone(), out_shape, out_dtype);

    cubecl::attention::launch_ref::<R>(
        &Strategy::Unit,
        client,
        &query.as_handle_ref(),
        &key.as_handle_ref(),
        &value.as_handle_ref(),
        &mask.as_ref().map(|mask| mask.as_handle_ref()),
        &out.as_handle_ref(),
        &AttentionElems {
            query_global: query.dtype.into(),
            query_tile: query.dtype.into(),
            key_global: key.dtype.into(),
            key_stage: key.dtype.into(),
            value_global: value.dtype.into(),
            value_stage: value.dtype.into(),
            key_value_tile: value.dtype.into(),
            softmax: query.dtype.into(),
            accumulator: out_dtype.into(),
            mask: mask.as_ref().map(|m| m.dtype).unwrap_or(DType::U8),
            out_global: out_dtype.into(),
            out_stage: out_dtype.into(),
        },
    )?;

    Ok(out)
}
