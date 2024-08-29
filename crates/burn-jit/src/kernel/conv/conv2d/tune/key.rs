use crate::tune::anchor;
use burn_tensor::{ops::ConvOptions, Shape};
use core::fmt::Debug;
use serde::{Deserialize, Serialize};
use std::{fmt::Display, hash::Hash};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
/// Autotune key representative of matmul versions
pub struct Conv2dAutotuneKey {
    kernel_size: [usize; 2],
    /// Stride.
    stride: [usize; 2],
    /// Padding.
    padding: [usize; 2],
    /// Dilation.
    dilation: [usize; 2],
    /// Groups.
    groups: usize,

    in_channels: usize,
    out_channels: usize,
    height: usize,
    width: usize,
    batch_size: usize,
}

impl Display for Conv2dAutotuneKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            format!(
                "Conv2d - Kernel:{:?} Stride:{:?} Padding:{:?} Dilation:{:?} Groups:{:?} In Channels:{:?} Out Channels:{:?} Batch:{:?}",
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                self.in_channels,
                self.out_channels,
                self.batch_size
            )
            .as_str(),
        )
    }
}

impl Conv2dAutotuneKey {
    /// Create a conv2d autotune key from the input shapes and options
    pub fn new(input: &Shape<4>, weights: &Shape<4>, options: &ConvOptions<2>) -> Self {
        let [batch_size, in_channels, height, width] = input.dims;
        let [out_channels, _, kernel_h, kernel_w] = weights.dims;

        Self {
            kernel_size: [kernel_h, kernel_w],
            stride: options.stride,
            padding: options.padding,
            dilation: options.dilation,
            groups: options.groups,
            in_channels: anchor(in_channels, None),
            height: anchor(height, None),
            width: anchor(width, None),
            out_channels: anchor(out_channels, None),
            batch_size: anchor(batch_size, None),
        }
    }
}
