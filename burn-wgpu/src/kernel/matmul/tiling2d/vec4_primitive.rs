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
    MatmulTiling2Dvec4PrimitiveRaw,
    "../../../template/matmul/blocktiling_2d/vec4_primitive.wgsl"
);

#[derive(new, Debug)]
struct MatmulTiling2Dvec4Primitive<E: WgpuElement> {
    b_m: usize,
    b_n: usize,
    b_k: usize,
    workgroup_size_x: usize,
    workgroup_size_y: usize,
    _elem: PhantomData<E>,
}

impl<E: WgpuElement> DynamicKernelSource for MatmulTiling2Dvec4Primitive<E> {
    fn source(self) -> SourceTemplate {
        MatmulTiling2Dvec4PrimitiveRaw::source()
            .register("b_m", self.b_m.to_string())
            .register("b_n", self.b_n.to_string())
            .register("b_k", self.b_k.to_string())
            .register("bm_x_bk_4", (self.b_m * self.b_k / 4).to_string())
            .register("bk_x_bn", (self.b_k * self.b_n).to_string())
            .register("workgroup_size_x", self.workgroup_size_x.to_string())
            .register("workgroup_size_y", self.workgroup_size_y.to_string())
            .register("workgroup_size_z", "1".to_string())
            .register("elem", E::type_name())
            .register("int", "i32")
    }

    fn id(&self) -> String {
        std::format!("{:?}", self)
    }
}

/// vec4 multiplication using tiling 2d algorithm with vec4 primitive with workgroups of size 16
pub fn matmul_tiling_2d_vec4_primitive_default<E: WgpuElement + Element, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    matmul_tiling_2d_vec4_primitive::<E, D>(lhs, rhs)
}

/// vec4 multiplication using tiling 2d algorithm with vec4 primitive with custom size workgroups
pub fn matmul_tiling_2d_vec4_primitive<E: WgpuElement + Element, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    let b_m = 64;
    let b_n = 64;
    let b_k = 32;
    let wgx = 16;
    let wgy = 16;
    let kernel = MatmulTiling2Dvec4Primitive::<E>::new(b_m, b_n, b_k, wgx, wgy);
    matmul_tiling_2d_launch(lhs, rhs, b_m, b_n, b_k, 4, 4, wgx, wgy, kernel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::matmul::utils::tests::{same_as_reference, same_as_reference_swapped_dims};

    #[test]
    pub fn test_matmul_vec4_primitive_straightforward() {
        test_with_params(1, 2, 1, 1, 1);
    }

    #[test]
    pub fn test_matmul_vec4_primitive_shapes_smaller_than_blocks() {
        test_with_params(8, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_vec4_primitive_n_smaller_than_m() {
        test_with_params(8, 8, 3, 1, 1);
    }

    #[test]
    pub fn test_matmul_vec4_primitive_m_smaller_than_n() {
        test_with_params(3, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_vec4_primitive_k_smaller_than_m_n() {
        test_with_params(8, 3, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_vec4_primitive_k_larger_than_m_n() {
        test_with_params(8, 48, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_vec4_primitive_multibatch_1_dim() {
        test_with_params(8, 8, 8, 3, 1);
    }

    #[test]
    pub fn test_matmul_vec4_primitive_multibatch_2_dims() {
        test_with_params(8, 8, 8, 3, 4);
    }

    #[test]
    pub fn test_matmul_vec4_primitive_blocks_divide_shapes_unevenly() {
        test_with_params(7, 7, 7, 1, 1);
    }

    #[test]
    pub fn test_matmul_vec4_primitive_medium() {
        test_with_params(17, 16, 16, 1, 1);
    }

    #[test]
    pub fn test_matmul_vec4_primitive_large() {
        test_with_params(134, 242, 250, 1, 1);
    }

    fn test_with_params(m: usize, k: usize, n: usize, batch_1: usize, batch_2: usize) {
        let func = matmul_tiling_2d_vec4_primitive::<f32, 4>;
        let shape_lhs = [batch_1, batch_2, m, k];
        let shape_rhs = [batch_1, batch_2, k, n];
        same_as_reference(func, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_tiling_2d_vec4_primitive_swapped_batches_no_padding() {
        let matmul_func = matmul_tiling_2d_vec4_primitive::<f32, 4>;
        let swap = [0, 1];
        let shape_lhs = [3, 2, 4, 4];
        let shape_rhs = [3, 2, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap, swap, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_tiling_2d_vec4_primitive_swapped_row_col_no_padding() {
        let matmul_func = matmul_tiling_2d_vec4_primitive::<f32, 4>;
        let swap_lhs = [0, 0];
        let swap_rhs = [2, 3];
        let shape_lhs = [3, 2, 4, 4];
        let shape_rhs = [3, 2, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_tiling_2d_vec4_primitive_swapped_row_with_batch_no_padding() {
        let matmul_func = matmul_tiling_2d_vec4_primitive::<f32, 4>;
        let swap_lhs = [0, 3];
        let swap_rhs = [0, 2];
        let shape_lhs = [4, 4, 4, 4];
        let shape_rhs = [4, 4, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
    }
}
