use std::marker::PhantomData;

use super::utils::shape_out;
use crate::{
    context::WorkGroup,
    element::WgpuElement,
    kernel::{build_info, into_contiguous, DynamicKernel, SourceTemplate, StaticKernel},
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

impl<E: WgpuElement> DynamicKernel for MatmulMemCoalescing<E> {
    fn source_template(self) -> SourceTemplate {
        MatmulMemCoalescingRaw::source_template()
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
) -> WgpuTensor<E, D> {
    matmul_mem_coalescing::<E, D>(lhs, rhs, 16, 16)
}

/// Matrix multiplication using memory coalescing algorithm with custom workgroup sizes
pub fn matmul_mem_coalescing<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
    workgroup_size_x: usize,
    workgroup_size_y: usize,
) -> WgpuTensor<E, D> {
    lhs.assert_is_on_same_device(&rhs);

    let lhs = into_contiguous(lhs);
    let rhs = into_contiguous(rhs);

    let shape_out = shape_out(&lhs, &rhs);
    let num_rows = lhs.shape.dims[D - 2];
    let num_cols = rhs.shape.dims[D - 1];

    let buffer = lhs
        .context
        .create_buffer(shape_out.num_elements() * core::mem::size_of::<E>());
    let output = WgpuTensor::new(lhs.context.clone(), shape_out, buffer);

    // set number of workgroups
    let blocks_needed_in_x = f32::ceil(num_rows as f32 / workgroup_size_x as f32) as u32;
    let blocks_needed_in_y = f32::ceil(num_cols as f32 / workgroup_size_y as f32) as u32;

    let kernel = lhs.context.compile_dynamic(MatmulMemCoalescing::<E>::new(
        workgroup_size_x,
        workgroup_size_y,
    ));

    let info = build_info(&[&lhs, &rhs, &output]);

    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output.shape.dims[i];
    }

    let workgroup = WorkGroup::new(blocks_needed_in_x, blocks_needed_in_y, num_iter as u32);

    lhs.context.execute(
        workgroup,
        kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    output
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
        let func = |lhs, rhs| {
            matmul_mem_coalescing::<f32, 4>(lhs, rhs, WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y)
        };
        let shape_lhs = [batch_1, batch_2, m, k];
        let shape_rhs = [batch_1, batch_2, k, n];
        same_as_reference(func, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_naive_swapped_batches_no_padding() {
        let matmul_func = |lhs, rhs| matmul_mem_coalescing::<f32, 4>(lhs, rhs, 2, 2);
        let swap = [0, 1];
        let shape_lhs = [3, 2, 4, 4];
        let shape_rhs = [3, 2, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap, swap, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_naive_swapped_row_col_no_padding() {
        let matmul_func = |lhs, rhs| matmul_mem_coalescing::<f32, 4>(lhs, rhs, 2, 2);
        let swap_lhs = [0, 0];
        let swap_rhs = [2, 3];
        let shape_lhs = [3, 2, 4, 4];
        let shape_rhs = [3, 2, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_naive_swapped_row_with_batch_no_padding() {
        let matmul_func = |lhs, rhs| matmul_mem_coalescing::<f32, 4>(lhs, rhs, 2, 2);
        let swap_lhs = [0, 3];
        let swap_rhs = [0, 2];
        let shape_lhs = [4, 4, 4, 4];
        let shape_rhs = [4, 4, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
    }
}
