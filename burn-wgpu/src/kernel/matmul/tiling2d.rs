use std::cmp::{max, min};

use crate::{
    context::WorkGroup,
    element::WgpuElement,
    kernel::{build_info, KernelSettings, SourceTemplate, StaticKernel},
    kernel_wgsl,
    tensor::WgpuTensor,
};
use burn_tensor::Shape;

const MAX_SHARED_MEMORY_SIZE: usize = 8192;

kernel_wgsl!(
    MatmulTiling2DRaw,
    "../../template/matmul_blocktiling_2d.wgsl"
);

struct MatmulTiling2D<
    const B_M: usize,
    const B_N: usize,
    const B_K: usize,
    const T_M: usize,
    const T_N: usize,
    const WORKGROUP_SIZE_X: usize,
    const WORKGROUP_SIZE_Y: usize,
>;

impl<
        const B_M: usize,
        const B_N: usize,
        const B_K: usize,
        const T_M: usize,
        const T_N: usize,
        const WORKGROUP_SIZE_X: usize,
        const WORKGROUP_SIZE_Y: usize,
    > StaticKernel for MatmulTiling2D<B_M, B_N, B_K, T_M, T_N, WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y>
{
    fn source_template() -> SourceTemplate {
        MatmulTiling2DRaw::source_template()
            .register("b_m", B_M.to_string())
            .register("b_n", B_N.to_string())
            .register("b_k", B_K.to_string())
            .register("bm_x_bk", (B_M * B_K).to_string())
            .register("bk_x_bn", (B_K * B_N).to_string())
            .register("t_m", T_M.to_string())
            .register("t_n", T_N.to_string())
            .register("tm_x_tn", (T_M * T_N).to_string())
    }
}

/// Matrix multiplication using tiling 2D algorithm with default parameters
pub fn matmul_tiling_2d_default<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    // Suppose a matmul of m1 of size [M, K] with m2 of size [K, N]
    // Block size along dim M
    const B_M: usize = 128;
    // // Block size along dim N
    const B_N: usize = 128;
    // // Block size along dim K
    const B_K: usize = 8;
    // // Tiling size along dim M
    const T_M: usize = 8;
    // // Tiling size along dim N
    const T_N: usize = 8;
    // WORKGROUP_SIZE_X = ceil(B_M / T_M)
    const WORKGROUP_SIZE_X: usize = 16;
    // WORKGROUP_SIZE_Y = ceil(B_N / T_N)
    const WORKGROUP_SIZE_Y: usize = 16;

    matmul_tiling_2d::<E, D, B_M, B_N, B_K, T_M, T_N, WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y>(lhs, rhs)
}

/// Matrix multiplication using tiling 2D algorithm with custom parameters
pub fn matmul_tiling_2d<
    E: WgpuElement,
    const D: usize,
    const B_M: usize,
    const B_N: usize,
    const B_K: usize,
    const T_M: usize,
    const T_N: usize,
    const WORKGROUP_SIZE_X: usize,
    const WORKGROUP_SIZE_Y: usize,
>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    assert!(B_K <= min(B_M, B_N), "B_K must be smaller than both B_M and B_M, otherwise there won't be enough threads to fill shared memory. ");
    assert!(B_K * max(B_M, B_N) <= MAX_SHARED_MEMORY_SIZE, "B_K x B_M and B_K x B_N must be smaller or equal than 8192, otherwise shared memory limit will be busted. ");
    assert!(
        WORKGROUP_SIZE_X == f32::ceil(B_M as f32 / T_M as f32) as usize,
        "Workgroup size x must equal ceil(B_M / T_M)"
    );
    assert!(
        WORKGROUP_SIZE_Y == f32::ceil(B_N as f32 / T_N as f32) as usize,
        "Workgroup size y must equal ceil(B_N / T_N)"
    );
    lhs.assert_is_on_same_device(&rhs);

    let mut shape_out = [0; D];
    lhs.shape
        .dims
        .iter()
        .zip(rhs.shape.dims.iter())
        .enumerate()
        .for_each(|(index, (dim_lhs, dim_rhs))| {
            shape_out[index] = usize::max(*dim_lhs, *dim_rhs);
        });

    let num_rows = lhs.shape.dims[D - 2];
    let num_cols = rhs.shape.dims[D - 1];
    shape_out[D - 2] = num_rows;
    shape_out[D - 1] = num_cols;
    let shape_out = Shape::new(shape_out);

    let buffer = lhs
        .context
        .create_buffer(shape_out.num_elements() * core::mem::size_of::<E>());
    let output = WgpuTensor::new(lhs.context.clone(), shape_out, buffer);

    // set number of workgroups
    let blocks_needed_in_x = f32::ceil(num_rows as f32 / B_M as f32) as u32;
    let blocks_needed_in_y = f32::ceil(num_cols as f32 / B_N as f32) as u32;

    let kernel = lhs.context.compile_static::<KernelSettings<
        MatmulTiling2D<B_M, B_N, B_K, T_M, T_N, WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y>,
        E,
        i32,
        WORKGROUP_SIZE_X,
        WORKGROUP_SIZE_Y,
        1,
    >>();

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
    use crate::tests::TestTensor;

    pub type ReferenceTensor<const D: usize> =
        burn_tensor::Tensor<burn_ndarray::NdArrayBackend<f32>, D>;

    #[test]
    pub fn test_matmul_tiling_2d_shapes_smaller_than_blocks() {
        test_with_params::<128, 128, 16, 8, 8, 16, 16>(8, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_m_not_equals_n() {
        test_with_params::<16, 16, 8, 8, 8, 2, 2>(8, 8, 3, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_k_smaller_than_m_n() {
        test_with_params::<16, 16, 8, 8, 8, 2, 2>(8, 3, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_k_larger_than_m_n() {
        test_with_params::<16, 16, 8, 8, 8, 2, 2>(8, 48, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_t_divides_b_unevenly() {
        test_with_params::<128, 128, 8, 7, 11, 19, 12>(8, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_small_parameters() {
        test_with_params::<128, 128, 8, 8, 8, 16, 16>(8, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_bm_not_equals_bn() {
        test_with_params::<32, 128, 8, 8, 8, 4, 16>(8, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_multibatch_1_dim() {
        test_with_params::<128, 128, 8, 8, 8, 16, 16>(8, 8, 8, 3, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_multibatch_2_dims() {
        test_with_params::<128, 128, 8, 8, 8, 16, 16>(8, 8, 8, 3, 4);
    }

    #[test]
    #[should_panic]
    pub fn test_matmul_tiling_2d_memory_busted_should_panic() {
        test_with_params::<128, 128, 128, 8, 8, 16, 16>(8, 8, 8, 1, 1);
    }

    #[test]
    #[should_panic]
    pub fn test_matmul_tiling_2d_bk_larger_than_bm_should_panic() {
        test_with_params::<64, 64, 128, 8, 8, 8, 8>(8, 8, 8, 1, 1);
    }

    #[test]
    #[should_panic]
    pub fn test_matmul_tiling_2d_workgroup_x_wrong_should_panic() {
        test_with_params::<128, 128, 16, 8, 8, 16, 8>(8, 8, 8, 1, 1);
    }

    #[test]
    #[should_panic]
    pub fn test_matmul_tiling_2d_workgroup_y_wrong_should_panic() {
        test_with_params::<128, 128, 16, 8, 8, 8, 7>(8, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_multiple_blocks() {
        test_with_params::<16, 16, 8, 8, 8, 2, 2>(32, 32, 32, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_k_bigger_than_bk() {
        test_with_params::<128, 128, 8, 8, 8, 16, 16>(8, 10, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_blocks_divide_shapes_unevenly() {
        test_with_params::<16, 16, 8, 8, 8, 2, 2>(31, 23, 17, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_large_parameters() {
        test_with_params::<256, 256, 16, 16, 16, 16, 16>(40, 40, 40, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_shapes_slightly_larger_than_blocks() {
        test_with_params::<32, 32, 8, 8, 8, 4, 4>(40, 40, 30, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_shapes_way_larger_than_blocks() {
        test_with_params::<16, 16, 8, 8, 8, 2, 2>(50, 50, 50, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_tm_larger_than_bm() {
        test_with_params::<2, 2, 2, 3, 2, 1, 1>(5, 5, 5, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_tn_larger_than_bn() {
        test_with_params::<2, 2, 2, 2, 3, 1, 1>(5, 5, 5, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_uneven_parameters() {
        test_with_params::<17, 15, 11, 13, 7, 2, 3>(24, 24, 24, 1, 1);
    }

    #[test]
    pub fn test_matmul_tiling_2d_uneven_parameters_2() {
        test_with_params::<11, 14, 10, 7, 17, 2, 1>(10, 24, 17, 1, 1);
    }

    fn test_with_params<
        const B_M: usize,
        const B_N: usize,
        const B_K: usize,
        const T_M: usize,
        const T_N: usize,
        const WORKGROUP_SIZE_X: usize,
        const WORKGROUP_SIZE_Y: usize,
    >(
        m: usize,
        k: usize,
        n: usize,
        batch_1: usize,
        batch_2: usize,
    ) {
        let func = |lhs, rhs| {
            matmul_tiling_2d::<f32, 4, B_M, B_N, B_K, T_M, T_N, WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y>(
                lhs, rhs,
            )
        };
        let shape_lhs = [batch_1, batch_2, m, k];
        let shape_rhs = [batch_1, batch_2, k, n];
        same_as_reference(func, shape_lhs, shape_rhs);
    }

    fn same_as_reference<F, const D: usize, S>(func: F, shape_lhs: S, shape_rhs: S)
    where
        F: Fn(WgpuTensor<f32, D>, WgpuTensor<f32, D>) -> WgpuTensor<f32, D>,
        S: Into<Shape<D>>,
    {
        let x = ReferenceTensor::random(shape_lhs, burn_tensor::Distribution::Uniform(-1.0, 1.0));
        let y = ReferenceTensor::random(shape_rhs, burn_tensor::Distribution::Uniform(-1.0, 1.0));

        let x_wgpu = TestTensor::from_data(x.to_data());
        let y_wgpu = TestTensor::from_data(y.to_data());

        let z_reference = x.matmul(y);

        let z = func(x_wgpu.into_primitive(), y_wgpu.into_primitive());
        let z = TestTensor::from_primitive(z);

        println!("{z}");
        z_reference.into_data().assert_approx_eq(&z.into_data(), 3);
    }
}
