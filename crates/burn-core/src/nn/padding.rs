use crate as burn;

use crate::tensor::ops::conv::calculate_conv_padding;

use crate::config::Config;

/// Padding configuration for 1D operators.
#[derive(Config, Debug, PartialEq)]
pub enum PaddingConfig1d {
    /// Dynamically calculate the amount of padding necessary to ensure that the output size will be
    /// the same as the input.
    Same,
    /// Same as no padding.
    Valid,
    /// Applies the specified amount of padding to all inputs.
    Explicit(usize),
}

impl PaddingConfig1d {
    pub(crate) fn calculate_padding_1d(
        &self,
        length: usize,
        kernel_size: usize,
        stride: usize,
    ) -> usize {
        let same_padding = || calculate_conv_padding(kernel_size, stride, length, length);
        match self {
            Self::Valid => 0,
            Self::Same => same_padding(),
            Self::Explicit(value) => *value,
        }
    }
}

/// Padding configuration for 2D operators.
#[derive(Config, Debug, PartialEq)]
pub enum PaddingConfig2d {
    /// Dynamically calculate the amount of padding necessary to ensure that the output size will be
    /// the same as the input.
    Same,
    /// Same as no padding.
    Valid,
    /// Applies the specified amount of padding to all inputs.
    Explicit(usize, usize),
}

impl PaddingConfig2d {
    pub(crate) fn calculate_padding_2d(
        &self,
        height: usize,
        width: usize,
        kernel_size: &[usize; 2],
        stride: &[usize; 2],
    ) -> [usize; 2] {
        let same_padding = || {
            let p1 = calculate_conv_padding(kernel_size[0], stride[0], height, height);
            let p2 = calculate_conv_padding(kernel_size[1], stride[1], width, width);

            [p1, p2]
        };

        match self {
            Self::Same => same_padding(),
            Self::Valid => [0, 0],
            Self::Explicit(v1, v2) => [*v1, *v2],
        }
    }
}

/// Padding configuration for 3D operators.
#[derive(Config, Debug, PartialEq)]
pub enum PaddingConfig3d {
    /// Dynamically calculate the amount of padding necessary to ensure that the output size will be
    /// the same as the input.
    Same,
    /// Same as no padding.
    Valid,
    /// Applies the specified amount of padding to all inputs.
    Explicit(usize, usize, usize),
}

impl PaddingConfig3d {
    pub(crate) fn calculate_padding_3d(
        &self,
        depth: usize,
        height: usize,
        width: usize,
        kernel_size: &[usize; 3],
        stride: &[usize; 3],
    ) -> [usize; 3] {
        let same_padding = || {
            let p1 = calculate_conv_padding(kernel_size[0], stride[0], depth, depth);
            let p2 = calculate_conv_padding(kernel_size[1], stride[1], height, height);
            let p3 = calculate_conv_padding(kernel_size[2], stride[2], width, width);

            [p1, p2, p3]
        };

        match self {
            Self::Same => same_padding(),
            Self::Valid => [0, 0, 0],
            Self::Explicit(v1, v2, v3) => [*v1, *v2, *v3],
        }
    }
}
