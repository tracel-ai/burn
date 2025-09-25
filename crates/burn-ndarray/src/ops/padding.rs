use crate::SharedArray;
use burn_tensor::Element;
use ndarray::{Array4, Array5};

use super::NdArrayOps;

pub(crate) fn apply_padding_4d<E: Element>(
    x: SharedArray<E>,
    padding: [usize; 2],
    elem: E,
) -> SharedArray<E> {
    let [batch_size, input_channels, height, width] = x.shape().try_into().unwrap();
    let [padding_height, padding_width] = padding;
    let padded_height = height + 2 * padding_height;
    let padded_width = width + 2 * padding_width;

    let x_new = Array4::from_elem(
        (batch_size, input_channels, padded_height, padded_width),
        elem,
    );
    let mut x_new = x_new.into_shared().into_dyn();

    x_new = NdArrayOps::slice_assign(
        x_new,
        &[
            burn_tensor::Slice::from(0..batch_size),
            burn_tensor::Slice::from(0..input_channels),
            burn_tensor::Slice::from(padding_height..height + padding_height),
            burn_tensor::Slice::from(padding_width..width + padding_width),
        ],
        x,
    );

    x_new
}

pub(crate) fn apply_padding_5d<E: Element>(
    x: SharedArray<E>,
    padding: [usize; 3],
    elem: E,
) -> SharedArray<E> {
    let [batch_size, input_channels, depth, height, width] = x.shape().try_into().unwrap();
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
    let mut x_new = x_new.into_shared().into_dyn();

    x_new = NdArrayOps::slice_assign(
        x_new,
        &[
            burn_tensor::Slice::from(0..batch_size),
            burn_tensor::Slice::from(0..input_channels),
            burn_tensor::Slice::from(padding_depth..depth + padding_depth),
            burn_tensor::Slice::from(padding_height..height + padding_height),
            burn_tensor::Slice::from(padding_width..width + padding_width),
        ],
        x,
    );

    x_new
}
