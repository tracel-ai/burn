use super::base::matmul_tiling_2d_launch;
use crate::{
    element::WgpuElement,
    kernel::{DynamicKernelSource, SourceTemplate, StaticKernelSource},
    matmul_tile_2d,
    tensor::WgpuTensor,
};

matmul_tile_2d!(
    MatmulTiling2DTileVectorized,
    "../../../template/matmul/blocktiling_2d/tile_vectorized.wgsl"
);
