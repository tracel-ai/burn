use burn_tensor::Element;

use crate::{
    compute::DynamicKernel,
    element::WgpuElement,
    kernel::{into_contiguous, DynamicKernelSource, SourceTemplate, StaticKernelSource},
    tensor::WgpuTensor,
};
use std::marker::PhantomData;

use crate::kernel_wgsl;

use super::base::{make_info_handle, make_workgroup, B_K, B_M, B_N, WORKGROUP_SIZE};

kernel_wgsl!(
    MatmulTiling2DUnpaddedRaw,
    "../../../template/matmul/blocktiling_2d/unpadded.wgsl"
);

#[derive(new, Debug)]
struct MatmulTiling2DUnpadded<E: WgpuElement> {
    _elem: PhantomData<E>,
}

impl<E: WgpuElement> DynamicKernelSource for MatmulTiling2DUnpadded<E> {
    fn source(&self) -> SourceTemplate {
        MatmulTiling2DUnpaddedRaw::source()
            .register("b_m", B_M.to_string())
            .register("b_n", B_N.to_string())
            .register("b_k", B_K.to_string())
            .register("bm_x_bk_4", (B_M * B_K / 4).to_string())
            .register("bk_x_bn_4", (B_K * B_N / 4).to_string())
            .register("workgroup_size_x", WORKGROUP_SIZE.to_string())
            .register("workgroup_size_y", WORKGROUP_SIZE.to_string())
            .register("workgroup_size_z", "1".to_string())
            .register("elem", E::type_name())
            .register("int", "i32")
    }

    fn id(&self) -> String {
        std::format!("{:?}", self)
    }
}

/// Matrix multiplication using tiling 2d algorithm with
/// vec4 primitive on both lhs and rhs, with no padding needed
pub fn matmul_tiling_2d_unpadded<E: WgpuElement + Element, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
    out: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    let lhs = match lhs.batch_swapped_with_row_col() {
        true => into_contiguous(lhs),
        false => lhs,
    };
    let rhs = match rhs.batch_swapped_with_row_col() {
        true => into_contiguous(rhs),
        false => rhs,
    };

    let workgroup = make_workgroup(&out.shape);
    let info_handle = make_info_handle(&lhs, &rhs, &out);

    lhs.client.execute(
        Box::new(DynamicKernel::new(
            MatmulTiling2DUnpadded::<E>::new(),
            workgroup,
        )),
        &[&lhs.handle, &rhs.handle, &out.handle, &info_handle],
    );

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::matmul::utils::tests::{same_as_reference, same_as_reference_swapped_dims};

    #[test]
    pub fn test_matmul_unpadded_straightforward() {
        test_with_params(1, 2, 1, 1, 1);
    }

    #[test]
    pub fn test_matmul_unpadded_shapes_smaller_than_blocks() {
        test_with_params(8, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_unpadded_shapes_equal_blocks() {
        test_with_params(64, 32, 64, 2, 2);
    }

    #[test]
    pub fn test_matmul_unpadded_m_exceeds_block() {
        test_with_params(75, 32, 64, 2, 2);
    }

    #[test]
    pub fn test_matmul_unpadded_k_exceeds_block() {
        test_with_params(64, 33, 32, 1, 1);
    }

    #[test]
    pub fn test_matmul_irregular_shape() {
        test_with_params(123, 255, 72, 3, 5);
    }

    #[test]
    pub fn test64_matmul_unpadded_n_exceeds_block() {
        test_with_params(64, 32, 75, 2, 2);
    }

    #[test]
    pub fn test_matmul_unpadded_n_smaller_than_m() {
        test_with_params(8, 8, 3, 1, 1);
    }

    #[test]
    pub fn test_matmul_unpadded_m_smaller_than_n() {
        test_with_params(3, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_unpadded_k_smaller_than_m_n() {
        test_with_params(8, 3, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_unpadded_k_larger_than_m_n() {
        test_with_params(8, 48, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_unpadded_multibatch_1_dim() {
        test_with_params(8, 8, 8, 3, 1);
    }

    #[test]
    pub fn test_matmul_unpadded_multibatch_2_dims() {
        test_with_params(8, 8, 8, 3, 4);
    }

    #[test]
    pub fn test_matmul_unpadded_blocks_divide_shapes_unevenly() {
        test_with_params(7, 7, 7, 1, 1);
    }

    #[test]
    pub fn test_matmul_unpadded_medium() {
        test_with_params(17, 16, 16, 1, 1);
    }

    #[test]
    pub fn test_matmul_unpadded_large() {
        test_with_params(134, 242, 250, 1, 1);
    }

    fn test_with_params(m: usize, k: usize, n: usize, batch_1: usize, batch_2: usize) {
        let func = matmul_tiling_2d_unpadded;
        let shape_lhs = [batch_1, batch_2, m, k];
        let shape_rhs = [batch_1, batch_2, k, n];
        same_as_reference(func, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_tiling_2d_primitive_swapped_batches_no_padding() {
        let matmul_func = matmul_tiling_2d_unpadded;
        let swap = [0, 1];
        let shape_lhs = [3, 2, 4, 4];
        let shape_rhs = [3, 2, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap, swap, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_tiling_2d_primitive_swapped_row_col_no_padding() {
        let matmul_func = matmul_tiling_2d_unpadded;
        let swap_lhs = [0, 0];
        let swap_rhs = [2, 3];
        let shape_lhs = [3, 2, 4, 4];
        let shape_rhs = [3, 2, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_tiling_2d_primitive_swapped_row_with_batch_no_padding() {
        let matmul_func = matmul_tiling_2d_unpadded;
        let swap_lhs = [0, 3];
        let swap_rhs = [0, 2];
        let shape_lhs = [4, 4, 4, 4];
        let shape_rhs = [4, 4, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
    }
}
