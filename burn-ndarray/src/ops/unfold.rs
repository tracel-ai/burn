use crate::{element::FloatNdArrayElement, tensor::NdArrayTensor};
use burn_tensor::ops::{ConvOptions, UnfoldOptions};
use ndarray::{Array3, Array4, Dim};

use super::conv::conv2d;

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
    let [batch_size, in_channels, in_height, in_width] = x.shape().dims;
    let stride = options.stride.unwrap_or([1, 1]);
    let padding = options.padding.unwrap_or([0, 0]);
    let dilation = options.dilation.unwrap_or([1, 1]);

    // See https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold for full explanation,
    // This calculates the number of patches with each patch having channels_out values
    let l_dim_1 =
        (in_height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1;
    let l_dim_2 =
        (in_width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1;
    let l = l_dim_1 * l_dim_2;
    let channels_out = in_channels * kernel_size[0] * kernel_size[1];

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

    let inner_array = unfolded.array;
    let mut reshaped_array = Array3::zeros([batch_size, channels_out, l]);

    // Iterate over each dimension and fill in the values from unfolded to reshaped
    for b in 0..batch_size {
        for c in 0..channels_out {
            let mut l_index = 0; // Index to keep track of where to put the value in the L dimension
            for h in 0..l_dim_1 {
                for w in 0..l_dim_2 {
                    // Copy the value from unfolded to reshaped
                    let value = inner_array[[b, c, h, w]];
                    reshaped_array[[b, c, l_index]] = value;

                    l_index += 1;
                }
            }
        }
    }

    // Convert the reshaped_array into the format that your NdArrayTensor needs
    let reshaped_array = reshaped_array.into_dyn().into_shared();
    NdArrayTensor::new(reshaped_array)
}
