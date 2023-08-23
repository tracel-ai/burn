use crate::{
    element::WgpuElement,
    kernel::{DynamicKernel, SourceTemplate, StaticKernel},
    matmul_tile_2d,
    tensor::WgpuTensor,
};

use super::base::matmul_tiling_2d_launch;

matmul_tile_2d!(
    MatmulTiling2DTile,
    "../../../template/matmul/blocktiling_2d/tile.wgsl"
);
