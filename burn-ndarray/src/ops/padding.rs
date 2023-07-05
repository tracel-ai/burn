use crate::{element::FloatNdArrayElement, tensor::NdArrayTensor, NdArrayBackend};
use burn_tensor::ops::TensorOps;
use ndarray::Array4;

pub(crate) fn apply_padding_4d<E: FloatNdArrayElement>(
    x: NdArrayTensor<E, 4>,
    padding: [usize; 2],
    elem: E,
) -> NdArrayTensor<E, 4> {
    let [batch_size, input_channels, height, width] = x.shape().dims;
    let [padding_height, padding_width] = padding;
    let padded_height = height + 2 * padding_height;
    let padded_width = width + 2 * padding_width;

    let x_new = Array4::from_elem(
        (batch_size, input_channels, padded_height, padded_width),
        elem,
    );
    let mut x_new = NdArrayTensor::new(x_new.into_shared().into_dyn());

    x_new = NdArrayBackend::slice_assign(
        x_new,
        [
            0..batch_size,
            0..input_channels,
            padding_height..(height + padding_height),
            padding_width..width + padding_width,
        ],
        x,
    );

    x_new
}
