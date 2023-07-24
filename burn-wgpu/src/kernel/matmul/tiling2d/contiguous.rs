use super::base::{matmul_tiling_2d_launch, register_template};
use crate::{
    element::WgpuElement,
    kernel::{KernelSettings, SourceTemplate, StaticKernel},
    matmul_tile_2d,
    tensor::WgpuTensor,
};

matmul_tile_2d!(
    MatmulTiling2DContiguous,
    "../../../template/matmul/blocktiling_2d/contiguous.wgsl"
);
