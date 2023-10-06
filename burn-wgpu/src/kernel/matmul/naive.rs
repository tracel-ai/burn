use super::utils::shape_out;
use crate::{
    compute::{StaticKernel, WorkGroup},
    element::WgpuElement,
    kernel::{build_info, into_contiguous, KernelSettings, SourceTemplate, StaticKernelSource},
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::WgpuTensor,
};

kernel_wgsl!(MatmulNaiveRaw, "../../template/matmul/naive.wgsl");

struct MatmulNaive<const WORKGROUP_SIZE_X: usize, const WORKGROUP_SIZE_Y: usize>;

impl<const WORKGROUP_SIZE_X: usize, const WORKGROUP_SIZE_Y: usize> StaticKernelSource
    for MatmulNaive<WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y>
{
    fn source() -> SourceTemplate {
        MatmulNaiveRaw::source()
            .register("block_size_m", WORKGROUP_SIZE_X.to_string())
            .register("block_size_n", WORKGROUP_SIZE_Y.to_string())
    }
}

/// Matrix multiplication using naive algorithm with workgroups of size 16
pub fn matmul_naive_default<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    matmul_naive::<E, D, 16, 16>(lhs, rhs)
}

/// Matrix multiplication using naive algorithm with custom workgroup sizes
pub fn matmul_naive<
    E: WgpuElement,
    const D: usize,
    const WORKGROUP_SIZE_X: usize,
    const WORKGROUP_SIZE_Y: usize,
>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    lhs.assert_is_on_same_device(&rhs);

    let lhs = into_contiguous(lhs);
    let rhs = into_contiguous(rhs);

    let shape_out = shape_out(&lhs, &rhs);

    let num_rows = lhs.shape.dims[D - 2];
    let num_cols = rhs.shape.dims[D - 1];

    let output = empty_device(lhs.client.clone(), lhs.device.clone(), shape_out);

    // set number of workgroups
    let blocks_needed_in_x = f32::ceil(num_rows as f32 / WORKGROUP_SIZE_X as f32) as u32;
    let blocks_needed_in_y = f32::ceil(num_cols as f32 / WORKGROUP_SIZE_Y as f32) as u32;
    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output.shape.dims[i];
    }
    let workgroup = WorkGroup::new(blocks_needed_in_x, blocks_needed_in_y, num_iter as u32);

    let kernel = StaticKernel::<
        KernelSettings<
            MatmulNaive<WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y>,
            E,
            i32,
            WORKGROUP_SIZE_X,
            WORKGROUP_SIZE_Y,
            1,
        >,
    >::new(workgroup);

    let info = build_info(&[&lhs, &rhs, &output]);

    let info_handle = lhs.client.create(bytemuck::cast_slice(&info));

    lhs.client.execute(
        Box::new(kernel),
        &[&lhs.handle, &rhs.handle, &output.handle, &info_handle],
    );

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::matmul::utils::tests::{same_as_reference, same_as_reference_swapped_dims};

    #[test]
    pub fn test_matmul_naive_straightforward() {
        test_with_params::<2, 2>(1, 2, 1, 1, 1);
    }

    #[test]
    pub fn test_matmul_naive_shapes_smaller_than_blocks() {
        test_with_params::<16, 16>(8, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_naive_n_smaller_than_m() {
        test_with_params::<2, 2>(8, 8, 3, 1, 1);
    }

    #[test]
    pub fn test_matmul_naive_m_smaller_than_n() {
        test_with_params::<2, 2>(3, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_naive_k_smaller_than_m_n() {
        test_with_params::<2, 2>(8, 3, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_naive_k_larger_than_m_n() {
        test_with_params::<2, 2>(8, 48, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_naive_multibatch_1_dim() {
        test_with_params::<2, 2>(8, 8, 8, 3, 1);
    }

    #[test]
    pub fn test_matmul_naive_multibatch_2_dims() {
        test_with_params::<2, 2>(8, 8, 8, 3, 4);
    }

    #[test]
    pub fn test_matmul_naive_blocks_divide_shapes_unevenly() {
        test_with_params::<3, 3>(7, 7, 7, 1, 1);
    }

    fn test_with_params<const WORKGROUP_SIZE_X: usize, const WORKGROUP_SIZE_Y: usize>(
        m: usize,
        k: usize,
        n: usize,
        batch_1: usize,
        batch_2: usize,
    ) {
        let func = matmul_naive::<f32, 4, WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y>;
        let shape_lhs = [batch_1, batch_2, m, k];
        let shape_rhs = [batch_1, batch_2, k, n];
        same_as_reference(func, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_naive_swapped_batches_no_padding() {
        let matmul_func = matmul_naive::<f32, 4, 2, 2>;
        let swap = [0, 1];
        let shape_lhs = [3, 2, 4, 4];
        let shape_rhs = [3, 2, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap, swap, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_naive_swapped_row_col_no_padding() {
        let matmul_func = matmul_naive::<f32, 4, 2, 2>;
        let swap_lhs = [0, 0];
        let swap_rhs = [2, 3];
        let shape_lhs = [3, 2, 4, 4];
        let shape_rhs = [3, 2, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_naive_swapped_row_with_batch_no_padding() {
        let matmul_func = matmul_naive::<f32, 4, 2, 2>;
        let swap_lhs = [0, 3];
        let swap_rhs = [0, 2];
        let shape_lhs = [4, 4, 4, 4];
        let shape_rhs = [4, 4, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
    }
}
