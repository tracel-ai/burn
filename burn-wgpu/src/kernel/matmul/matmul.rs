use crate::{element::WgpuElement, kernel::tile::matmul_tiling_2d_default, tensor::WgpuTensor};

/// Public matmul
pub fn matmul<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    matmul_tiling_2d_default(lhs, rhs)
}
