// Based off MatmulTiling2DContiguousVectorized (contiguous_vectorized.rs)
// as it is the fastest on my macbook

use super::base::matmul_tiling_2d_launch;
use crate::{
    element::WgpuElement,
    kernel::{DynamicKernelSource, SourceTemplate, StaticKernelSource},
    matmul_tile_2d,
    tensor::WgpuTensor,
};

matmul_tile_2d!(
    MatmulTiling2DMatrixPrimitive,
    "../../../template/matmul/blocktiling_2d/matrix_primitive.wgsl"
);
