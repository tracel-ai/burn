use candle_core::{DType, Device, Shape, Tensor};

use crate::element::CandleElement;

pub(crate) fn fill<E: CandleElement, S: Into<Shape>>(
    value: E,
    shape: S,
    dtype: DType,
    device: &Device,
) -> Tensor {
    let values = (Tensor::ones((1), dtype, device).unwrap() * value.elem::<f64>()).unwrap();
    values.expand(shape).unwrap()
}

pub(crate) fn fill_like<E: CandleElement>(value: E, reference_tensor: &Tensor) -> Tensor {
    fill(
        value,
        reference_tensor.shape(),
        reference_tensor.dtype(),
        reference_tensor.device(),
    )
}

/// Broadcasts two tensors to a common shape for comparison operations
pub(crate) fn broadcast_for_comparison(
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<(Tensor, Tensor), candle_core::Error> {
    let broadcast_shape = lhs
        .shape()
        .broadcast_shape_binary_op(rhs.shape(), "comparison")?;

    let lhs = if broadcast_shape != *lhs.shape() {
        lhs.broadcast_as(&broadcast_shape)?
    } else {
        lhs.clone()
    };

    let rhs = if broadcast_shape != *rhs.shape() {
        rhs.broadcast_as(&broadcast_shape)?
    } else {
        rhs.clone()
    };

    Ok((lhs, rhs))
}
