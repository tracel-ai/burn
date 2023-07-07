use crate::{
    element::WgpuElement,
    kernel::{KernelSettings, SourceTemplate, StaticKernel},
    matmul_tile_2d,
    tensor::WgpuTensor,
};

use super::base::{matmul_tiling_2d_launch, register_template};

matmul_tile_2d!(
    MatmulTiling2DTile,
    "../../../template/matmul/blocktiling_2d/tile.wgsl"
);

// #[cfg(test)]
// mod tests {
//     use burn_tensor::Shape;
//
//     use super::*;
//     use crate::tests::TestTensor;
//
//     pub type ReferenceTensor<const D: usize> =
//         burn_tensor::Tensor<burn_ndarray::NdArrayBackend<f32>, D>;
//
//     #[test]
//     pub fn test_matmul_tiling_2d_shapes_smaller_than_blocks() {
//         test_with_params::<128, 128, 16, 8, 8, 16, 16>(8, 8, 8, 1, 1);
//     }
//
//     #[test]
//     pub fn test_matmul_tiling_2d_m_not_equals_n() {
//         test_with_params::<16, 16, 8, 8, 8, 2, 2>(16, 8, 16, 1, 1);
//     }
//
//     #[test]
//     pub fn test_matmul_tiling_2d_k_smaller_than_m_n() {
//         test_with_params::<16, 16, 4, 8, 8, 2, 2>(16, 4, 16, 1, 1);
//     }
//
//     #[test]
//     pub fn test_matmul_tiling_2d_k_larger_than_m_n() {
//         test_with_params::<8, 8, 8, 8, 8, 1, 1>(8, 48, 8, 1, 1);
//     }
//
//     #[test]
//     #[should_panic]
//     pub fn test_matmul_tiling_2d_t_divides_b_unevenly_should_panic() {
//         test_with_params::<128, 128, 8, 7, 11, 19, 12>(8, 8, 8, 1, 1);
//     }
//
//     #[test]
//     pub fn test_matmul_tiling_2d_bm_not_equals_bn() {
//         test_with_params::<2, 4, 2, 2, 4, 1, 1>(8, 8, 8, 1, 1);
//     }
//
//     #[test]
//     pub fn test_matmul_tiling_2d_multibatch_1_dim() {
//         test_with_params::<8, 8, 8, 8, 8, 1, 1>(8, 8, 8, 3, 1);
//     }
//
//     #[test]
//     pub fn test_matmul_tiling_2d_multibatch_2_dims() {
//         test_with_params::<8, 8, 8, 8, 8, 1, 1>(8, 8, 8, 3, 4);
//     }
//
//     #[test]
//     #[should_panic]
//     pub fn test_matmul_tiling_2d_memory_busted_should_panic() {
//         test_with_params::<128, 128, 128, 8, 8, 16, 16>(8, 8, 8, 1, 1);
//     }
//
//     #[test]
//     #[should_panic]
//     pub fn test_matmul_tiling_2d_bk_larger_than_bm_should_panic() {
//         test_with_params::<64, 64, 128, 8, 8, 8, 8>(8, 8, 8, 1, 1);
//     }
//
//     #[test]
//     #[should_panic]
//     pub fn test_matmul_tiling_2d_workgroup_x_wrong_should_panic() {
//         test_with_params::<128, 128, 16, 8, 8, 16, 8>(8, 8, 8, 1, 1);
//     }
//
//     #[test]
//     #[should_panic]
//     pub fn test_matmul_tiling_2d_workgroup_y_wrong_should_panic() {
//         test_with_params::<128, 128, 16, 8, 8, 8, 7>(8, 8, 8, 1, 1);
//     }
//
//     #[test]
//     pub fn test_matmul_tiling_2d_multiple_blocks() {
//         test_with_params::<16, 16, 8, 8, 8, 2, 2>(32, 32, 32, 1, 1);
//     }
//
//     #[test]
//     pub fn test_matmul_tiling_2d_k_bigger_than_bk() {
//         test_with_params::<8, 8, 8, 8, 8, 1, 1>(8, 16, 8, 1, 1);
//     }
//
//     #[test]
//     pub fn test_matmul_tiling_2d_blocks_divide_shapes_unevenly() {
//         test_with_params::<16, 16, 8, 8, 8, 2, 2>(31, 23, 17, 1, 1);
//     }
//
//     #[test]
//     pub fn test_matmul_tiling_2d_shapes_way_larger_than_blocks() {
//         test_with_params::<16, 16, 8, 8, 8, 2, 2>(48, 48, 48, 1, 1);
//     }
//
//     #[test]
//     #[should_panic]
//     pub fn test_matmul_tiling_2d_tm_larger_than_bm_should_panic() {
//         test_with_params::<2, 2, 2, 3, 2, 1, 1>(5, 5, 5, 1, 1);
//     }
//
//     #[test]
//     #[should_panic]
//     pub fn test_matmul_tiling_2d_tn_larger_than_bn_should_panic() {
//         test_with_params::<2, 2, 2, 2, 3, 1, 1>(5, 5, 5, 1, 1);
//     }
//
//     #[test]
//     #[should_panic]
//     pub fn test_matmul_tiling_2d_uneven_parameters_should_panic() {
//         test_with_params::<17, 15, 11, 13, 7, 2, 3>(24, 24, 24, 1, 1);
//     }
//
//     #[test]
//     #[should_panic]
//     pub fn test_matmul_tiling_2d_uneven_parameters_2_should_panic() {
//         test_with_params::<11, 14, 10, 7, 17, 2, 1>(10, 24, 17, 1, 1);
//     }
//
//     fn test_with_params<
//         const B_M: usize,
//         const B_N: usize,
//         const B_K: usize,
//         const T_M: usize,
//         const T_N: usize,
//         const WORKGROUP_SIZE_X: usize,
//         const WORKGROUP_SIZE_Y: usize,
//     >(
//         m: usize,
//         k: usize,
//         n: usize,
//         batch_1: usize,
//         batch_2: usize,
//     ) {
//         let func = |lhs, rhs| {
//             matmul_tiling_2d::<f32, 4, B_M, B_N, B_K, T_M, T_N, WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y>(
//                 lhs, rhs,
//             )
//         };
//         let shape_lhs = [batch_1, batch_2, m, k];
//         let shape_rhs = [batch_1, batch_2, k, n];
//         same_as_reference(func, shape_lhs, shape_rhs);
//     }
//
//     fn same_as_reference<F, const D: usize, S>(func: F, shape_lhs: S, shape_rhs: S)
//     where
//         F: Fn(WgpuTensor<f32, D>, WgpuTensor<f32, D>) -> WgpuTensor<f32, D>,
//         S: Into<Shape<D>>,
//     {
//         let x = ReferenceTensor::random(shape_lhs, burn_tensor::Distribution::Uniform(-1.0, 1.0));
//         let y = ReferenceTensor::random(shape_rhs, burn_tensor::Distribution::Uniform(-1.0, 1.0));
//
//         let x_wgpu = TestTensor::from_data(x.to_data());
//         let y_wgpu = TestTensor::from_data(y.to_data());
//
//         let z_reference = x.matmul(y);
//
//         let z = func(x_wgpu.into_primitive(), y_wgpu.into_primitive());
//         let z = TestTensor::from_primitive(z);
//
//         println!("{z_reference}");
//         println!("{z}");
//         z_reference.into_data().assert_approx_eq(&z.into_data(), 3);
//     }
// }
