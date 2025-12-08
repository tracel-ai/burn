use crate::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_tensor::{DType, Shape};
use cubecl::attention::{
    Strategy,
    components::{AttentionSetupError, AttentionStorageTypes},
};

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
    let dtypes = AttentionStorageTypes {
        query: query.dtype.into(),
        key: key.dtype.into(),
        value: value.dtype.into(),
        mask: mask.as_ref().map(|m| m.dtype).unwrap_or(DType::U8).into(),
        out: out.dtype.into(),
    };

    cubecl::attention::launch_ref::<R>(
        &Strategy::Unit(cubecl::attention::kernels::SharedAttentionSettings {
            tiling_scheme: None,
            reuse_key_value: false,
            two_rows_in_array_tile: false,
        }),
        client,
        &query.as_handle_ref(),
        &key.as_handle_ref(),
        &value.as_handle_ref(),
        &mask.as_ref().map(|mask| mask.as_handle_ref()),
        &out.as_handle_ref(),
        dtypes,
    )?;

    Ok(out)
}
