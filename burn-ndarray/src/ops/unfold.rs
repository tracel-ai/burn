use crate::{element::FloatNdArrayElement, tensor::NdArrayTensor};
use burn_tensor::{
    ops::{ConvOptions, UnfoldOptions},
    Shape,
};
use ndarray::{Array4, Dim};

use super::{conv::conv2d, NdArrayOps};

/// Constructs a special weight tensor used for unfolding.
///
/// # Notes
/// The idea behind using convolution for unfolding is to leverage the sliding window mechanism of
/// convolution. By creating a weight tensor with ones in a particular pattern, we are able to borrow
/// the convolution operation's mechanism as it moves across the input tensor, picking up the desired
/// values in the pattern of the unfolding operation.
fn create_unfolding_weight<E: FloatNdArrayElement>(
    in_channels: usize,
    kernel_size: [usize; 2],
) -> NdArrayTensor<E, 4> {
    let mut weight = Array4::zeros(Dim([
        in_channels * kernel_size[0] * kernel_size[1],
        in_channels,
        kernel_size[0],
        kernel_size[1],
    ]));

    for k in 0..in_channels {
        for i in 0..kernel_size[0] {
            for j in 0..kernel_size[1] {
                let output_channel = k * kernel_size[0] * kernel_size[1] + i * kernel_size[1] + j;
                weight[[output_channel, k, i, j]] = E::one();
            }
        }
    }

    NdArrayTensor::new(weight.into_dyn().into_shared())
}

pub(crate) fn unfold4d<E: FloatNdArrayElement>(
    x: NdArrayTensor<E, 4>,
    kernel_size: [usize; 2],
    options: UnfoldOptions,
) -> NdArrayTensor<E, 3> {
    let [_batch_size, in_channels, _in_height, _in_width] = x.shape().dims;
    let stride = options.stride.unwrap_or([1, 1]);
    let padding = options.padding.unwrap_or([0, 0]);
    let dilation = options.dilation.unwrap_or([1, 1]);

    let weight = create_unfolding_weight(in_channels, kernel_size);
    let unfolded = conv2d(
        x,
        weight,
        None,
        ConvOptions {
            stride,
            padding,
            dilation,
            groups: 1,
        },
    );

    let [batch_size, channels_out, out_height, out_width] = unfolded.shape().dims;

    NdArrayOps::reshape(
        unfolded,
        Shape::new([batch_size, channels_out, out_height * out_width]),
    )
}
