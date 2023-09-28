use burn_tensor::Element;

use crate::{
    element::WgpuElement,
    kernel::{DynamicKernelSource, SourceTemplate, StaticKernelSource},
    tensor::WgpuTensor,
};
use std::marker::PhantomData;

use crate::kernel_wgsl;

use super::base::matmul_tiling_2d_launch;

kernel_wgsl!(
    MatmulTiling2DMatrixPrimitiveRaw,
    "../../../template/matmul/blocktiling_2d/matrix_primitive.wgsl"
);

#[derive(new, Debug)]
struct MatmulTiling2DMatrixPrimitive<E: WgpuElement> {
    _elem: PhantomData<E>,
}

impl<E: WgpuElement> DynamicKernelSource for MatmulTiling2DMatrixPrimitive<E> {
    fn source(self) -> SourceTemplate {
        MatmulTiling2DMatrixPrimitiveRaw::source()
            .register("elem", E::type_name())
            .register("int", "i32")
    }

    fn id(&self) -> String {
        std::format!("{:?}", self)
    }
}

/// Matrix multiplication using tiling 2d algorithm with matrix primitive with workgroups of size 16
pub fn matmul_tiling_2d_matrix_primitive_default<E: WgpuElement + Element, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    matmul_tiling_2d_matrix_primitive::<E, D>(lhs, rhs)
}

/// Matrix multiplication using tiling 2d algorithm with matrix primitive with custom size workgroups
pub fn matmul_tiling_2d_matrix_primitive<E: WgpuElement + Element, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    let kernel = MatmulTiling2DMatrixPrimitive::<E>::new();
    matmul_tiling_2d_launch(lhs, rhs, 64, 64, 32, 4, 4, 16, 16, kernel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::matmul::utils::tests::{same_as_reference, same_as_reference_swapped_dims};

    #[test]
    pub fn test_matmul_matrix_primitive_straightforward() {
        test_with_params(1, 2, 1, 1, 1);
    }

    #[test]
    pub fn test_matmul_matrix_primitive_shapes_smaller_than_blocks() {
        test_with_params(8, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_matrix_primitive_n_smaller_than_m() {
        test_with_params(8, 8, 3, 1, 1);
    }

    #[test]
    pub fn test_matmul_matrix_primitive_m_smaller_than_n() {
        test_with_params(3, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_matrix_primitive_k_smaller_than_m_n() {
        test_with_params(8, 3, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_matrix_primitive_k_larger_than_m_n() {
        test_with_params(8, 48, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_matrix_primitive_multibatch_1_dim() {
        test_with_params(8, 8, 8, 3, 1);
    }

    #[test]
    pub fn test_matmul_matrix_primitive_multibatch_2_dims() {
        test_with_params(8, 8, 8, 3, 4);
    }

    #[test]
    pub fn test_matmul_matrix_primitive_blocks_divide_shapes_unevenly() {
        test_with_params(7, 7, 7, 1, 1);
    }

    #[test]
    pub fn test_matmul_matrix_primitive_medium() {
        test_with_params(31, 31, 31, 1, 1);
    }

    #[test]
    pub fn test_matmul_matrix_primitive_large() {
        test_with_params(34, 34, 34, 1, 1);
    }

    fn test_with_params(m: usize, k: usize, n: usize, batch_1: usize, batch_2: usize) {
        let func = |lhs, rhs| matmul_tiling_2d_matrix_primitive::<f32, 4>(lhs, rhs);
        let shape_lhs = [batch_1, batch_2, m, k];
        let shape_rhs = [batch_1, batch_2, k, n];
        same_as_reference(func, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_tiling_2d_matrix_primitive_swapped_batches_no_padding() {
        let matmul_func = |lhs, rhs| matmul_tiling_2d_matrix_primitive::<f32, 4>(lhs, rhs);
        let swap = [0, 1];
        let shape_lhs = [3, 2, 4, 4];
        let shape_rhs = [3, 2, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap, swap, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_tiling_2d_matrix_primitive_swapped_row_col_no_padding() {
        let matmul_func = |lhs, rhs| matmul_tiling_2d_matrix_primitive::<f32, 4>(lhs, rhs);
        let swap_lhs = [0, 0];
        let swap_rhs = [2, 3];
        let shape_lhs = [3, 2, 4, 4];
        let shape_rhs = [3, 2, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_tiling_2d_matrix_primitive_swapped_row_with_batch_no_padding() {
        let matmul_func = |lhs, rhs| matmul_tiling_2d_matrix_primitive::<f32, 4>(lhs, rhs);
        let swap_lhs = [0, 3];
        let swap_rhs = [0, 2];
        let shape_lhs = [4, 4, 4, 4];
        let shape_rhs = [4, 4, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
    }
}
