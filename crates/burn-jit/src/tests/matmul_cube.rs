#[burn_tensor_testgen::testgen(matmul_cube)]
mod tests {
    use super::*;
    use burn_jit::kernel::matmul::tiling2d_cube::{
        compute_loop_tests, load_shared_memory_tests, outer_product_tests, write_output_tests,
    };
    use burn_jit::kernel::matmul::{matmul, MatmulStrategy, Tiling2dConfig};
    use burn_tensor::{Shape, Tensor};

    #[test]
    pub fn tiling2d_matmul_outer_product_vectorized_test() {
        outer_product_tests::tile_outer_product_vectorized_unit_test::<TestRuntime>(
            &Default::default(),
        )
    }

    #[test]
    pub fn tiling2d_matmul_outer_product_scalar_test() {
        outer_product_tests::tile_outer_product_scalar_unit_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn tiling2d_matmul_compute_loop_vectorized_test() {
        compute_loop_tests::compute_loop_unit_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn tiling2d_matmul_read_whole_vectorized_test() {
        load_shared_memory_tests::read_whole_unit_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn tiling2d_matmul_read_partial_vectorized_test() {
        load_shared_memory_tests::read_partial_unit_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn tiling2d_matmul_read_zeros_vectorized_test() {
        load_shared_memory_tests::read_zeros_unit_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn tiling2d_matmul_load_tile_no_checks_unit_test() {
        load_shared_memory_tests::load_tile_no_checks_unit_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn tiling2d_matmul_load_tile_vertical_checks_unit_test() {
        load_shared_memory_tests::load_tile_vertical_checks_unit_test::<TestRuntime>(
            &Default::default(),
        )
    }

    #[test]
    pub fn tiling2d_matmul_load_tile_horizontal_checks_unit_test() {
        load_shared_memory_tests::load_tile_horizontal_checks_unit_test::<TestRuntime>(
            &Default::default(),
        )
    }

    #[test]
    pub fn write_tile_plain_unit_test() {
        load_shared_memory_tests::write_tile_plain_unit_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn write_tile_transposed_unit_test() {
        load_shared_memory_tests::write_tile_transposed_unit_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn load_lhs_transposed_unit_test() {
        load_shared_memory_tests::load_lhs_transposed_unit_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn load_rhs_plain_unit_test() {
        load_shared_memory_tests::load_rhs_plain_unit_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn write_results_inner_loop_unit_test() {
        write_output_tests::write_results_inner_loop_unit_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn write_results_to_output_unit_test() {
        write_output_tests::write_results_to_output_unit_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn write_results_to_output_partial_unit_test() {
        write_output_tests::write_results_to_output_partial_unit_test::<TestRuntime>(
            &Default::default(),
        )
    }

    #[test]
    pub fn write_to_output_over_height_unit_test() {
        write_output_tests::write_to_output_over_height_unit_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn write_to_output_over_width_unit_test() {
        write_output_tests::write_to_output_over_width_unit_test::<TestRuntime>(&Default::default())
    }
}
