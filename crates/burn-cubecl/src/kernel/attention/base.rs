use burn_tensor::{DType, Shape};
use cubek::attention::{
    Strategy,
    components::{AttentionSetupError, AttentionStorageTypes},
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

    cubek::attention::launch_ref::<R>(
        &Strategy::Unit(Default::default()),
        client,
        &query.as_handle_ref(),
        &key.as_handle_ref(),
        &value.as_handle_ref(),
        &mask.as_ref().map(|mask| mask.as_handle_ref()),
        &out.as_handle_ref(),
        AttentionStorageTypes {
            query: query.dtype.into(),
            key: key.dtype.into(),
            value: value.dtype.into(),
            mask: mask.as_ref().map(|m| m.dtype).unwrap_or(DType::U8).into(),
            out: out_dtype.into(),
        },
    )?;

    Ok(out)
}
