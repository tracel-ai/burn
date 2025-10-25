use super::{ConvOptions, UnfoldOptions};
use crate::backend::Backend;
use crate::ops::FloatTensor;
use crate::{ElementConversion, Shape, TensorData, TensorMetadata};
use alloc::vec;
use alloc::vec::Vec;

/// Constructs a special weight tensor used for unfolding.
///
/// # Notes
///
/// The idea behind using convolution for unfolding is to leverage the sliding window mechanism of
/// convolution. By creating a weight tensor with ones in a particular pattern, we are able to borrow
/// the convolution operation's mechanism as it moves across the input tensor, picking up the desired
/// values in the pattern of the unfolding operation.
pub(crate) fn create_unfolding_weight<B: Backend>(
    in_channels: usize,
    kernel_size: [usize; 2],
    device: &B::Device,
) -> FloatTensor<B> {
    let shape = Shape::new([
        in_channels * kernel_size[0] * kernel_size[1],
        in_channels,
        kernel_size[0],
        kernel_size[1],
    ]);

    let mut strides = [0; 4];
    let mut current = 1;
    shape
        .dims
        .iter()
        .enumerate()
        .rev()
        .for_each(|(index, val)| {
            strides[index] = current;
            current *= val;
        });

    let num_elements = shape.num_elements();

    let mut weight: Vec<B::FloatElem> = vec![0.0.elem(); num_elements];

    for k in 0..in_channels {
        for i in 0..kernel_size[0] {
            for j in 0..kernel_size[1] {
                let output_channel = k * kernel_size[0] * kernel_size[1] + i * kernel_size[1] + j;
                let index =
                    output_channel * strides[0] + k * strides[1] + i * strides[2] + j * strides[3];

                weight[index] = 1.elem();
            }
        }
    }

    B::float_from_data(TensorData::new(weight, shape), device)
}

/// Compute the unfold4d operation using the conv2d operations.
pub(crate) fn unfold4d_using_conv2d<B: Backend>(
    x: FloatTensor<B>,
    kernel_size: [usize; 2],
    options: UnfoldOptions,
) -> FloatTensor<B> {
    let [_batch_size, in_channels, _in_height, _in_width] = x.shape().dims();
    let weight = create_unfolding_weight::<B>(in_channels, kernel_size, &B::float_device(&x));
    let unfolded = B::conv2d(
        x,
        weight,
        None,
        ConvOptions::new(options.stride, options.padding, options.dilation, 1),
    );

    let [batch_size, channels_out, out_height, out_width] = unfolded.shape().dims();

    B::float_reshape(
        unfolded,
        Shape::new([batch_size, channels_out, out_height * out_width]),
    )
}

/// Calculate the number of unfolding windows that can be extracted from a dimension of given size.
pub fn calculate_unfold_windows(dim_size: usize, window_size: usize, step_size: usize) -> usize {
    assert!(step_size > 0);
    let x = dim_size + step_size;
    if x < window_size {
        0
    } else {
        (x - window_size) / step_size
    }
}

/// Calculate the output shape for an unfold operation.
///
/// The operation yields a view with all complete windows of size `size` in dimension `dim`;
/// where windows are advanced by `step` at each index.
///
/// The number of windows is `max(0, (shape[dim] - size).ceil_div(step))`.
///
/// # Arguments
///
/// * `shape` - The input shape to unfold; of shape ``[pre=..., dim shape, post=...]``
/// * `dim` - the dimension to unfold.
/// * `size` - the size of each unfolded window.
/// * `step` - the step between each window.
///
/// # Returns
///
/// A shape with ``[pre=..., windows, post=..., size]``.
pub fn calculate_unfold_shape<S: Into<Shape>>(
    shape: S,
    dim: usize,
    size: usize,
    step: usize,
) -> Vec<usize> {
    let mut shape = shape.into().to_vec();
    let d_shape = shape[dim];
    let windows = calculate_unfold_windows(d_shape, size, step);
    shape[dim] = windows;
    shape.push(size);

    shape
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_unfold_windows() {
        assert_eq!(calculate_unfold_windows(2, 5, 1), 0);

        assert_eq!(calculate_unfold_windows(2, 3, 1), 0);
        assert_eq!(calculate_unfold_windows(3, 3, 1), 1);
        assert_eq!(calculate_unfold_windows(4, 3, 1), 2);
        assert_eq!(calculate_unfold_windows(5, 3, 1), 3);

        assert_eq!(calculate_unfold_windows(2, 3, 2), 0);
        assert_eq!(calculate_unfold_windows(3, 3, 2), 1);
        assert_eq!(calculate_unfold_windows(4, 3, 2), 1);
        assert_eq!(calculate_unfold_windows(5, 3, 2), 2);
    }

    #[test]
    fn test_calculate_unfold_shape() {
        assert_eq!(calculate_unfold_shape([2, 6, 6], 1, 3, 2), vec![2, 2, 6, 3]);
    }
}
