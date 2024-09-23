use crate::{
    element::{FloatNdArrayElement, QuantElement},
    tensor::NdArrayTensor,
    NdArray,
};
use burn_tensor::ops::FloatTensorOps;
use ndarray::{Array4, Array5};

pub(crate) fn apply_padding_4d<E: FloatNdArrayElement, Q: QuantElement>(
    x: NdArrayTensor<E>,
    padding: [usize; 2],
    elem: E,
) -> NdArrayTensor<E> {
    let [batch_size, input_channels, height, width] = x.shape().dims();
    let [padding_height, padding_width] = padding;
    let padded_height = height + 2 * padding_height;
    let padded_width = width + 2 * padding_width;

    let x_new = Array4::from_elem(
        (batch_size, input_channels, padded_height, padded_width),
        elem,
    );
    let mut x_new = NdArrayTensor::new(x_new.into_shared().into_dyn());

    x_new = NdArray::<E, Q>::float_slice_assign(
        x_new,
        &[
            0..batch_size,
            0..input_channels,
            padding_height..height + padding_height,
            padding_width..width + padding_width,
        ],
        x,
    );

    x_new
}

pub(crate) fn apply_padding_5d<E: FloatNdArrayElement, Q: QuantElement>(
    x: NdArrayTensor<E>,
    padding: [usize; 3],
    elem: E,
) -> NdArrayTensor<E> {
    let [batch_size, input_channels, depth, height, width] = x.shape().dims();
    let [padding_depth, padding_height, padding_width] = padding;
    let padded_depth = depth + 2 * padding_depth;
    let padded_height = height + 2 * padding_height;
    let padded_width = width + 2 * padding_width;

    let x_new = Array5::from_elem(
        (
            batch_size,
            input_channels,
            padded_depth,
            padded_height,
            padded_width,
        ),
        elem,
    );
    let mut x_new = NdArrayTensor::new(x_new.into_shared().into_dyn());

    x_new = NdArray::<E, Q>::float_slice_assign(
        x_new,
        &[
            0..batch_size,
            0..input_channels,
            padding_depth..depth + padding_depth,
            padding_height..height + padding_height,
            padding_width..width + padding_width,
        ],
        x,
    );

    x_new
}
