use crate::{element::WgpuElement, kernel::tile::matmul_tiling_2d, tensor::WgpuTensor};

/// Public matmul
pub fn matmul<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    const B_M: usize = 64;
    // Block size along dim N
    const B_N: usize = 64;
    // Block size along dim K
    const B_K: usize = 32;
    // Tiling size along dim M
    const T_M: usize = 4;
    // Tiling size along dim N
    const T_N: usize = 4;
    // WORKGROUP_SIZE_X = ceil(B_M / T_M)
    const WORKGROUP_SIZE_X: usize = B_M / T_M;
    // WORKGROUP_SIZE_Y = ceil(B_N / T_N)
    const WORKGROUP_SIZE_Y: usize = B_N / T_N;

    matmul_tiling_2d::<E, D, B_M, B_N, B_K, T_M, T_N, WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y>(lhs, rhs)
}
