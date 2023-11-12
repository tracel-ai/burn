use burn_tensor::Shape;
use std::marker::PhantomData;

use crate::{
    compute::{DynamicKernel, Kernel, WorkGroup},
    element::WgpuElement,
    kernel::{
        build_info, into_contiguous, DynamicKernelSource, SourceTemplate, StaticKernelSource,
        WORKGROUP_DEFAULT,
    },
    kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(
    MatmulMemCoalescingRaw,
    "../../template/matmul/mem_coalescing.wgsl"
);

#[derive(new, Debug)]
struct MatmulMemCoalescing<E: WgpuElement> {
    workgroup_size_x: usize,
    workgroup_size_y: usize,
    _elem: PhantomData<E>,
}

impl<E: WgpuElement> DynamicKernelSource for MatmulMemCoalescing<E> {
    fn source(&self) -> SourceTemplate {
        MatmulMemCoalescingRaw::source()
            .register("workgroup_size_x", self.workgroup_size_x.to_string())
            .register("workgroup_size_y", self.workgroup_size_y.to_string())
            .register("elem", E::type_name())
            .register("int", "i32")
    }

    fn id(&self) -> String {
        std::format!("{:?}", self)
    }
}

/// Matrix multiplication using memory coalescing algorithm with workgroups of size 16
pub fn matmul_mem_coalescing_default<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
    out: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    matmul_mem_coalescing::<E, D>(lhs, rhs, out, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT)
}

/// Matrix multiplication using memory coalescing algorithm with custom workgroup sizes
pub fn matmul_mem_coalescing<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
    output: WgpuTensor<E, D>,
    workgroup_size_x: usize,
    workgroup_size_y: usize,
) -> WgpuTensor<E, D> {
    lhs.assert_is_on_same_device(&rhs);

    let lhs = into_contiguous(lhs);
    let rhs = into_contiguous(rhs);

    let info = build_info(&[&lhs, &rhs, &output]);

    let info_handle = lhs.client.create(bytemuck::cast_slice(&info));

    let kernel = matmul_mem_coalescing_kernel::<E, D>(
        &lhs.shape,
        &rhs.shape,
        &output.shape,
        workgroup_size_x,
        workgroup_size_y,
    );

    lhs.client.execute(
        kernel,
        &[&lhs.handle, &rhs.handle, &output.handle, &info_handle],
    );

    output
}

fn matmul_mem_coalescing_kernel<E: WgpuElement, const D: usize>(
    lhs_shape: &Shape<D>,
    rhs_shape: &Shape<D>,
    output_shape: &Shape<D>,
    workgroup_size_x: usize,
    workgroup_size_y: usize,
) -> Box<dyn Kernel> {
    let num_rows = lhs_shape.dims[D - 2];
    let num_cols = rhs_shape.dims[D - 1];

    // set number of workgroups
    let blocks_needed_in_x = f32::ceil(num_rows as f32 / workgroup_size_x as f32) as u32;
    let blocks_needed_in_y = f32::ceil(num_cols as f32 / workgroup_size_y as f32) as u32;
    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output_shape.dims[i];
    }

    let workgroup = WorkGroup::new(blocks_needed_in_x, blocks_needed_in_y, num_iter as u32);

    Box::new(DynamicKernel::new(
        MatmulMemCoalescing::<E>::new(workgroup_size_x, workgroup_size_y),
        workgroup,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::matmul::utils::tests::{same_as_reference, same_as_reference_swapped_dims};

    #[test]
    pub fn test_matmul_mem_coalescing_straightforward() {
        test_with_params::<2, 2>(1, 2, 1, 1, 1);
    }

    #[test]
    pub fn test_matmul_mem_coalescing_shapes_smaller_than_blocks() {
        test_with_params::<16, 16>(8, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_mem_coalescing_n_smaller_than_m() {
        test_with_params::<2, 2>(8, 8, 3, 1, 1);
    }

    #[test]
    pub fn test_matmul_mem_coalescing_m_smaller_than_n() {
        test_with_params::<2, 2>(3, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_mem_coalescing_k_smaller_than_m_n() {
        test_with_params::<2, 2>(8, 3, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_mem_coalescing_k_larger_than_m_n() {
        test_with_params::<2, 2>(8, 48, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_mem_coalescing_multibatch_1_dim() {
        test_with_params::<2, 2>(8, 8, 8, 3, 1);
    }

    #[test]
    pub fn test_matmul_mem_coalescing_multibatch_2_dims() {
        test_with_params::<2, 2>(8, 8, 8, 3, 4);
    }

    #[test]
    pub fn test_matmul_mem_coalescing_blocks_divide_shapes_unevenly() {
        test_with_params::<3, 3>(7, 7, 7, 1, 1);
    }

    fn test_with_params<const WORKGROUP_SIZE_X: usize, const WORKGROUP_SIZE_Y: usize>(
        m: usize,
        k: usize,
        n: usize,
        batch_1: usize,
        batch_2: usize,
    ) {
        let func = |lhs, rhs, out| {
            matmul_mem_coalescing::<f32, 4>(lhs, rhs, out, WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
        };
        let shape_lhs = [batch_1, batch_2, m, k];
        let shape_rhs = [batch_1, batch_2, k, n];
        same_as_reference(func, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_naive_swapped_batches_no_padding() {
        let matmul_func = |lhs, rhs, out| matmul_mem_coalescing::<f32, 4>(lhs, rhs, out, 2, 2);
        let swap = [0, 1];
        let shape_lhs = [3, 2, 4, 4];
        let shape_rhs = [3, 2, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap, swap, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_naive_swapped_row_col_no_padding() {
        let matmul_func = |lhs, rhs, out| matmul_mem_coalescing::<f32, 4>(lhs, rhs, out, 2, 2);
        let swap_lhs = [0, 0];
        let swap_rhs = [2, 3];
        let shape_lhs = [3, 2, 4, 4];
        let shape_rhs = [3, 2, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_naive_swapped_row_with_batch_no_padding() {
        let matmul_func = |lhs, rhs, out| matmul_mem_coalescing::<f32, 4>(lhs, rhs, out, 2, 2);
        let swap_lhs = [0, 3];
        let swap_rhs = [0, 2];
        let shape_lhs = [4, 4, 4, 4];
        let shape_rhs = [4, 4, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
    }
}
