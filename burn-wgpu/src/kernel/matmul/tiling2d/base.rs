use crate::{
    context::{Context, WorkGroup},
    element::WgpuElement,
    kernel::{build_info, into_contiguous, matmul::utils::shape_out},
    tensor::WgpuTensor,
};
use burn_tensor::Shape;
use std::{
    cmp::{max, min},
    sync::Arc,
};
use wgpu::ComputePipeline;

use super::padding::{crop, pad_round, PaddingOutput};

const MAX_SHARED_MEMORY_SIZE: usize = 8192;

pub(super) fn empty_from_context<E: WgpuElement, const D: usize>(
    context: Arc<Context>,
    shape: &Shape<D>,
) -> WgpuTensor<E, D> {
    let buffer = context.create_buffer(shape.num_elements() * core::mem::size_of::<E>());

    WgpuTensor::new(context, shape.clone(), buffer)
}

/// Create a source template for tile 2d matmul.
#[macro_export(local_inner_macros)]
macro_rules! matmul_tile_2d {
    (
        $struct:ident,
        $file:expr
    ) => {
        matmul_tile_2d!(
            $struct,
            $file,
            B_M 64,
            B_N 64,
            B_K 32,
            T_M 4,
            T_N 4
        );
    };

    (
        $struct:ident,
        $file:expr,
        B_M $bm:expr,
        B_N $bn:expr,
        B_K $bk:expr,
        T_M $tm:expr,
        T_N $tn:expr
     ) => {
        #[derive(new, Debug)]
        struct $struct<E: WgpuElement> {
            b_m: usize,
            b_n: usize,
            b_k: usize,
            t_m: usize,
            t_n: usize,
            workgroup_size_x: usize,
            workgroup_size_y: usize,
            _elem: core::marker::PhantomData<E>,
        }

        impl<E: WgpuElement> DynamicKernel for $struct<E> {
            fn source_template(self) -> SourceTemplate {
                kernel_wgsl!(Raw, $file);

                Raw::source_template()
                    .register("b_m", self.b_m.to_string())
                    .register("b_n", self.b_n.to_string())
                    .register("b_k", self.b_k.to_string())
                    .register("bm_x_bk", (self.b_m * self.b_k).to_string())
                    .register("bk_x_bn", (self.b_k * self.b_n).to_string())
                    .register("t_m", self.t_m.to_string())
                    .register("t_n", self.t_n.to_string())
                    .register("tm_x_tn", (self.t_m * self.t_n).to_string())
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

        /// Matrix multiplication using tiling 2D algorithm with default parameters
        pub fn matmul_tiling_2d_default<E: WgpuElement, const D: usize>(
            lhs: WgpuTensor<E, D>,
            rhs: WgpuTensor<E, D>,
        ) -> WgpuTensor<E, D> {
            // Suppose a matmul of m1 of size [M, K] with m2 of size [K, N]
            // Block size along dim M
            const B_M: usize = $bm;
            // // Block size along dim N
            const B_N: usize = $bn;
            // // Block size along dim K
            const B_K: usize = $bk;
            // // Tiling size along dim M
            const T_M: usize = $tm;
            // // Tiling size along dim N
            const T_N: usize = $tn;
            // WORKGROUP_SIZE_X = ceil(B_M / T_M)
            const WORKGROUP_SIZE_X: usize = B_M / T_M;
            // WORKGROUP_SIZE_Y = ceil(B_N / T_N)
            const WORKGROUP_SIZE_Y: usize = B_N / T_N;

            matmul_tiling_2d(
                lhs, rhs,
                B_M,
                B_N,
                B_K,
                T_N,
                T_N,
                WORKGROUP_SIZE_X,
                WORKGROUP_SIZE_Y,
            )
        }

        /// Matrix multiplication using tiling 2D algorithm with custom parameters
        #[allow(clippy::too_many_arguments)]
        pub fn matmul_tiling_2d<
            E: WgpuElement,
            const D: usize,
       >(
            lhs: WgpuTensor<E, D>,
            rhs: WgpuTensor<E, D>,
            b_m: usize,
            b_n: usize,
            b_k: usize,
            t_m: usize,
            t_n: usize,
            workgroup_size_x: usize,
            workgroup_size_y: usize,

        ) -> WgpuTensor<E, D> {
            let kernel = $struct::<E>::new(b_m, b_n, b_k, t_m, t_n, workgroup_size_x, workgroup_size_y);
            let kernel = lhs.context.compile_dynamic(kernel);
            matmul_tiling_2d_launch::<
                E,
                D,
            >(
                lhs,
                rhs,
                kernel,
                b_m,
                b_n,
                b_k,
                t_m,
                t_n,
                workgroup_size_x,
                workgroup_size_y,
              )
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            use $crate::kernel::matmul::utils::tests::same_as_reference;
            use $crate::kernel::matmul::utils::tests::same_as_reference_swapped_dims;

            #[test]
            pub fn test_matmul_tiling_2d_large_blocks() {
                test_with_params::<128, 128, 8, 4, 4, 32, 32>(8, 8, 8, 1, 1);
            }

            #[test]
            pub fn test_matmul_tiling_2d_m_larger_than_n() {
                test_with_params::<64, 64, 32, 4, 4, 16, 16>(64, 32, 4, 1, 1);
            }

            #[test]
            pub fn test_matmul_tiling_2d_n_larger_than_m() {
                test_with_params::<64, 64, 32, 4, 4, 16, 16>(4, 32, 64, 1, 1);
            }

            #[test]
            pub fn test_matmul_tiling_2d_shapes_smaller_than_blocks() {
                test_with_params::<64, 64, 8, 4, 4, 16, 16>(8, 8, 8, 1, 1);
            }

            #[test]
            pub fn test_matmul_tiling_2d_m_not_equals_n() {
                test_with_params::<16, 16, 8, 2, 2, 8, 8>(16, 8, 16, 1, 1);
            }

            #[test]
            pub fn test_matmul_tiling_2d_k_smaller_than_m_n() {
                test_with_params::<16, 16, 4, 2, 2, 8, 8>(16, 4, 16, 1, 1);
            }

            #[test]
            pub fn test_matmul_tiling_2d_k_larger_than_m_n() {
                test_with_params::<8, 8, 8, 2, 2, 4, 4>(8, 48, 8, 1, 1);
            }

            #[test]
            #[should_panic]
            pub fn test_matmul_tiling_2d_t_divides_b_unevenly_should_panic() {
                test_with_params::<128, 128, 8, 7, 11, 19, 12>(8, 8, 8, 1, 1);
            }

            #[test]
            pub fn test_matmul_tiling_2d_bm_not_equals_bn() {
                test_with_params::<8, 16, 8, 2, 4, 4, 4>(8, 8, 16, 1, 1);
            }

            #[test]
            pub fn test_matmul_tiling_2d_multibatch_1_dim() {
                test_with_params::<8, 8, 8, 2, 2, 4, 4>(8, 8, 8, 3, 1);
            }

            #[test]
            pub fn test_matmul_tiling_2d_multibatch_2_dims() {
                test_with_params::<8, 8, 8, 2, 2, 4, 4>(8, 8, 8, 3, 4);
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
                test_with_params::<16, 16, 8, 2, 2, 8, 8>(32, 32, 32, 1, 1);
            }

            #[test]
            pub fn test_matmul_tiling_2d_k_bigger_than_bk() {
                test_with_params::<8, 8, 8, 2, 2, 4, 4>(8, 16, 8, 1, 1);
            }

            #[test]
            pub fn test_matmul_tiling_2d_blocks_divide_shapes_unevenly() {
                test_with_params::<16, 16, 8, 2, 2, 8, 8>(31, 23, 17, 1, 1);
            }

            #[test]
            pub fn test_matmul_tiling_2d_shapes_way_larger_than_blocks() {
                test_with_params::<16, 16, 8, 2, 2, 8, 8>(48, 48, 48, 1, 1);
            }

            #[test]
            #[should_panic]
            pub fn test_matmul_tiling_2d_tm_larger_than_bm_should_panic() {
                test_with_params::<2, 2, 2, 3, 2, 1, 1>(5, 5, 5, 1, 1);
            }

            #[test]
            #[should_panic]
            pub fn test_matmul_tiling_2d_tn_larger_than_bn_should_panic() {
                test_with_params::<2, 2, 2, 2, 3, 1, 1>(5, 5, 5, 1, 1);
            }

            #[test]
            #[should_panic]
            pub fn test_matmul_tiling_2d_uneven_parameters_should_panic() {
                test_with_params::<17, 15, 11, 13, 7, 2, 3>(24, 24, 24, 1, 1);
            }

            #[test]
            #[should_panic]
            pub fn test_matmul_tiling_2d_uneven_parameters_2_should_panic() {
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
                    matmul_tiling_2d::<f32, 4, >(
                        lhs, rhs,
                        B_M, B_N, B_K, T_M, T_N, WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y
                    )
                };
                let shape_lhs = [batch_1, batch_2, m, k];
                let shape_rhs = [batch_1, batch_2, k, n];
                same_as_reference(func, shape_lhs, shape_rhs);
            }

            #[test]
            fn test_matmul_tiling_2d_swapped_batches_no_padding() {
                const DIM: usize = 4;

                let matmul_func = |lhs, rhs| {
                    matmul_tiling_2d::<f32, 4, >(
                        lhs, rhs,
                        DIM, DIM, DIM, 2, 2, 2, 2
                    )
                };
                let swap = [0, 1];
                let shape_lhs = [3, 2, DIM, DIM];
                let shape_rhs = [3, 2, DIM, DIM];
                same_as_reference_swapped_dims(matmul_func, swap, swap, shape_lhs, shape_rhs);
            }


            #[test]
            fn test_matmul_tiling_2d_swapped_row_col_no_padding() {
                const DIM: usize = 4;

                let matmul_func = |lhs, rhs| {
                    matmul_tiling_2d::<f32, 4, >(
                        lhs, rhs,
                        DIM, DIM, DIM, 2, 2, 2, 2
                    )
                };
                let swap_lhs = [0, 0];
                let swap_rhs = [2, 3];
                let shape_lhs = [3, 2, DIM, DIM];
                let shape_rhs = [3, 2, DIM, DIM];
                same_as_reference_swapped_dims(matmul_func, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
            }

            #[test]
            fn test_matmul_tiling_2d_swapped_row_with_batch_no_padding() {
                const DIM: usize = 4;

                let matmul_func = |lhs, rhs| {
                    matmul_tiling_2d::<f32, 4, >(
                        lhs, rhs,
                        DIM, DIM, DIM, 2, 2, 2, 2
                    )
                };
                let swap_lhs = [0, 3];
                let swap_rhs = [0, 2];
                let shape_lhs = [DIM, DIM, DIM, DIM];
                let shape_rhs = [DIM, DIM, DIM, DIM];
                same_as_reference_swapped_dims(matmul_func, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
            }
        }
    };
}

#[allow(clippy::too_many_arguments)]
pub(super) fn matmul_parameter_assertions<E: WgpuElement, const D: usize>(
    b_m: usize,
    b_n: usize,
    b_k: usize,
    t_m: usize,
    t_n: usize,
    workgroup_size_x: usize,
    workgroup_size_y: usize,
    lhs: &WgpuTensor<E, D>,
    rhs: &WgpuTensor<E, D>,
) {
    assert!(b_k <= min(b_m, b_n), "B_K must be smaller than both B_M and B_M, otherwise there won't be enough threads to fill shared memory. ");
    assert!(b_k * max(b_m, b_n) <= MAX_SHARED_MEMORY_SIZE, "B_K x B_M and B_K x B_N must be smaller or equal than 8192, otherwise shared memory limit will be busted. ");
    assert!(
        b_m % t_m == 0 && b_n % t_n == 0,
        "T_M must divide B_M in this version"
    );
    assert!(
        workgroup_size_x == b_m / t_m,
        "Workgroup size x must equal B_M / T_M"
    );
    assert!(
        workgroup_size_y == b_n / t_n,
        "Workgroup size y must equal B_N / T_N"
    );
    lhs.assert_is_on_same_device(rhs);
}

pub(super) fn make_workgroup<const D: usize>(
    output_shape: Shape<D>,
    b_m: usize,
    b_n: usize,
) -> WorkGroup {
    let num_blocks_x = f32::ceil(output_shape.dims[D - 2] as f32 / b_m as f32) as u32;
    let num_blocks_y = f32::ceil(output_shape.dims[D - 1] as f32 / b_n as f32) as u32;
    let mut num_blocks_z = 1;
    for i in 0..D - 2 {
        num_blocks_z *= output_shape.dims[i];
    }

    WorkGroup::new(num_blocks_x, num_blocks_y, num_blocks_z as u32)
}

pub(super) fn make_info_buffers<E: WgpuElement, const D: usize>(
    lhs: &WgpuTensor<E, D>,
    rhs: &WgpuTensor<E, D>,
    output: &WgpuTensor<E, D>,
) -> Arc<wgpu::Buffer> {
    let info = build_info(&[lhs, rhs, output]);
    rhs.context
        .create_buffer_with_data(bytemuck::cast_slice(&info))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn matmul_tiling_2d_launch<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
    kernel: Arc<ComputePipeline>,
    b_m: usize,
    b_n: usize,
    b_k: usize,
    t_m: usize,
    t_n: usize,
    workgroup_size_x: usize,
    workgroup_size_y: usize,
) -> WgpuTensor<E, D> {
    matmul_parameter_assertions::<E, D>(
        b_m,
        b_n,
        b_k,
        t_m,
        t_n,
        workgroup_size_x,
        workgroup_size_y,
        &lhs,
        &rhs,
    );

    let final_output_shape = shape_out(&lhs, &rhs);

    // A tensor may need to be padded, in which case it will implicitly become contiguous
    // If not needed, it is only turned into contiguous if some batch dim has been swapped with row or col dim.
    // If batches were swapped among themselves, or if the last two dims are transposed, the underlying
    // kernel handles it without needing to turn it into contiguous.
    let round_lhs = pad_round(lhs, b_m, b_k);
    let lhs = match round_lhs {
        PaddingOutput::Unchanged(tensor) if tensor.batch_swapped_with_row_col() => {
            into_contiguous(tensor)
        }
        _ => round_lhs.into_tensor(),
    };
    let round_rhs = pad_round(rhs, b_k, b_n);
    let rhs = match round_rhs {
        PaddingOutput::Unchanged(tensor) if tensor.batch_swapped_with_row_col() => {
            into_contiguous(tensor)
        }
        _ => round_rhs.into_tensor(),
    };

    let rounded_output_shape = shape_out(&lhs, &rhs);

    let output = empty_from_context::<E, D>(rhs.context.clone(), &rounded_output_shape);

    let workgroup = make_workgroup(rounded_output_shape, b_m, b_n);
    let info_buffers = make_info_buffers(&lhs, &rhs, &output);

    lhs.context.execute(
        workgroup,
        kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    crop(output, final_output_shape)
}
