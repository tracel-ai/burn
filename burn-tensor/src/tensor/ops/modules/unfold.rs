use crate::backend::Backend;
use crate::{Data, ElementConversion, Shape};
use alloc::vec;
use alloc::vec::Vec;

use super::{ConvOptions, UnfoldOptions};

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
) -> B::TensorPrimitive<4> {
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

    B::from_data(Data::new(weight, shape), device)
}

/// Compute the unfold4d operation using the conv2d operations.
pub(crate) fn unfold4d_using_conv2d<B: Backend>(
    x: B::TensorPrimitive<4>,
    kernel_size: [usize; 2],
    options: UnfoldOptions,
) -> B::TensorPrimitive<3> {
    let [_batch_size, in_channels, _in_height, _in_width] = B::shape(&x).dims;
    let weight = create_unfolding_weight::<B>(in_channels, kernel_size, &B::device(&x));
    let unfolded = B::conv2d(
        x,
        weight,
        None,
        ConvOptions {
            stride: options.stride,
            padding: options.padding,
            dilation: options.dilation,
            groups: 1,
        },
    );

    let [batch_size, channels_out, out_height, out_width] = B::shape(&unfolded).dims;

    B::reshape(
        unfolded,
        Shape::new([batch_size, channels_out, out_height * out_width]),
    )
}
