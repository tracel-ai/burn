use burn_backend::DType;
use cubecl::AutotuneKey;
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of matmul versions
pub struct ConvAutotuneKey {
    pub kernel_size: Vec<usize>,
    pub stride: Vec<usize>,
    pub padding: Vec<(usize, usize)>,
    pub dilation: Vec<usize>,
    pub groups: usize,
    #[autotune(anchor)]
    pub in_channels: usize,
    #[autotune(anchor)]
    pub out_channels: usize,
    pub shape: Vec<usize>,
    #[autotune(anchor)]
    pub batch_size: usize,
    pub has_bias: bool,
    pub dtype: DType,

    pub lhs_shape_align: u8,
    pub lhs_stride_align: u8,
    pub rhs_shape_align: u8,
    pub rhs_stride_align: u8,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of matmul versions
pub struct ConvTranspose2dAutotuneKey {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub padding_out: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
    #[autotune(anchor)]
    pub in_channels: usize,
    #[autotune(anchor)]
    pub out_channels: usize,
    #[autotune(anchor)]
    pub height: usize,
    #[autotune(anchor)]
    pub width: usize,
    #[autotune(anchor)]
    pub batch_size: usize,
    pub has_bias: bool,
    pub dtype: DType,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(padding: Vec<(usize, usize)>) -> ConvAutotuneKey {
        ConvAutotuneKey {
            kernel_size: vec![3, 3],
            stride: vec![1, 1],
            padding,
            dilation: vec![1, 1],
            groups: 1,
            in_channels: 4,
            out_channels: 8,
            shape: vec![16, 16],
            batch_size: 2,
            has_bias: false,
            dtype: DType::F32,
            lhs_shape_align: 1,
            lhs_stride_align: 1,
            rhs_shape_align: 1,
            rhs_stride_align: 1,
        }
    }

    #[test]
    fn convolution_key_distinguishes_end_padding() {
        assert_ne!(key(vec![(0, 1), (0, 0)]), key(vec![(0, 2), (0, 0)]));
    }
}
