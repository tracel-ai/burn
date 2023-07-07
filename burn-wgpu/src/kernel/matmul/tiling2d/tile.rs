use crate::{
    element::WgpuElement,
    kernel::{KernelSettings, SourceTemplate, StaticKernel},
    matmul_tile_2d,
    tensor::WgpuTensor,
};

use super::base::{matmul_tiling_2d_launch, register_template};

matmul_tile_2d!(
    MatmulTiling2DTile,
    "../../../template/matmul/blocktiling_2d/tile.wgsl"
);
