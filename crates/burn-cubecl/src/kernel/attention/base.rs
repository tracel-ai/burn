use burn_tensor::{DType, Shape};
use cubecl::attention::{Strategy, components::AttentionSetupError};

use crate::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};

pub fn attention<R: CubeRuntime>(
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

    // strategy: &Strategy,
    // client: &ComputeClient<R>,
    // query: &TensorHandleRef<R>,
    // key: &TensorHandleRef<R>,
    // value: &TensorHandleRef<R>,
    // mask: &Option<TensorHandleRef<R>>,
    // out: &TensorHandleRef<R>,
    // attention_elems: &AttentionElems,
    cubecl::attention::launch_ref::<R>(
        Strategy::BlackboxAccelerated,
        client,
        query.as_handle_ref(),
        key.as_handle_ref(),
        value.as_handle_ref(),
        mask.map(|mask| mask.as_handle_ref()),
        out.as_handle_ref(),
        out_dtype,
    )?;

    Ok(out)
}
