use crate::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::{DType, Shape};
use cubek::attention::{
    definition::{
        AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions, AttentionSetupError,
    },
    launch::Strategy,
};

/// Launch a flash attention kernel
pub fn flash_attention<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    is_causal: bool,
    out_dtype: DType,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    let client = &query.client;
    let device = &query.device;

    let num_batches = query.meta.shape[0];
    let num_heads = query.meta.shape[1];
    let seq_q = query.meta.shape[2];
    let val_dim = value.meta.shape[3];
    let out_shape = Shape::new([num_batches, num_heads, seq_q, val_dim]);

    let out = empty_device_dtype::<R>(client.clone(), device.clone(), out_shape, out_dtype);
    let dtypes = AttentionGlobalTypes {
        query: query.dtype.into(),
        key: key.dtype.into(),
        value: value.dtype.into(),
        mask: mask.as_ref().map(|m| m.dtype).unwrap_or(DType::U8).into(),
        out: out.dtype.into(),
    };

    cubek::attention::launch::launch_ref::<R>(
        Strategy::Unit(cubek::attention::launch::BlueprintStrategy::Inferred(())),
        client,
        &query.as_handle_ref(),
        &key.as_handle_ref(),
        &value.as_handle_ref(),
        &mask.as_ref().map(|mask| mask.as_handle_ref()),
        &out.as_handle_ref(),
        &dtypes,
        AttentionOptions {
            causal: is_causal,
            accumulator_precision: AccumulatorPrecision::Strict(cubecl::ir::StorageType::Scalar(
                cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
            )),
        },
    )?;

    Ok(out)
}
