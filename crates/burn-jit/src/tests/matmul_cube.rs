use burn_cube::prelude::*;
use burn_tensor::backend::Backend;

use crate::{
    kernel::matmul::{
        computation_loop, computation_loop_expand, CubeTiling2dConfig, Tiling2dState,
    },
    tensor::JitTensor,
};

#[cube(launch)]
fn computation_loop_call<F: Float>(
    lhs: Tensor<F>,
    rhs: Tensor<F>,
    out: Tensor<F>,
    config: Comptime<CubeTiling2dConfig>,
) {
    // let kernel_state = Tiling2dState {
    //     n_loops,
    //     k,
    //     lhs,
    //     rhs,
    //     out,
    //     offset_lhs,
    //     offset_rhs,
    //     offset_output,
    //     row,
    //     col,
    //     dim_m,
    //     dim_k,
    //     dim_n,
    //     unit_col,
    //     unit_row,
    //     shared_lhs,
    //     shared_rhs,
    //     register_m,
    //     register_n,
    //     results,
    //     lhs_stride_col,
    //     lhs_stride_row,
    //     rhs_stride_col,
    //     rhs_stride_row,
    //     out_stride_row,
    //     out_stride_col,
    // };
    // computation_loop(kernel_state, config);
}

pub fn matmul_tiling_2d_cube<B: Backend>() {
    let cube_count = CubeCount { x: 1, y: 1, z: 1 };
    let vectorization_factor = 1;
    let settings = KernelSettings::default()
        .vectorize_input(0, vectorization_factor as u8)
        .vectorize_input(1, vectorization_factor as u8)
        .vectorize_output(0, vectorization_factor as u8)
        .cube_dim(CubeDim { x: 1, y: 1, z: 1 });
    computation_loop_call_launch::<F32, TestRuntime>(
        client,
        cube_count,
        settings,
        TensorHandle::<TestRuntime>::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
        TensorHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
        TensorHandle::new(&out.handle, &out.strides, &out.shape.dims),
        CubeTiling2dConfig::new(config, m, k, n, vectorization_factor as usize),
    );
}

#[burn_tensor_testgen::testgen(matmul_cube)]
mod tests {
    use super::*;
    use burn_jit::kernel::matmul::{matmul, MatmulStrategy, Tiling2dConfig};
    use burn_tensor::{Shape, Tensor};

    #[test]
    pub fn tiling2d_matmul_computation_loop_test() {
        burn_jit::tests::matmul_cube::matmul_tiling_2d_cube::<TestBackend>()
    }
}
