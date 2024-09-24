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
