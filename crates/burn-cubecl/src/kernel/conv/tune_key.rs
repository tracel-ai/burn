use burn_backend::DType;
use cubecl::AutotuneKey;
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of matmul versions
pub struct ConvAutotuneKey {
    pub kernel_size: Vec<usize>,
    pub stride: Vec<usize>,
    pub padding: Vec<usize>,
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
