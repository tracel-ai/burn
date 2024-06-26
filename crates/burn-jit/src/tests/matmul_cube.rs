#[burn_tensor_testgen::testgen(matmul_cube)]
mod tests {
    use super::*;
    use burn_jit::kernel::matmul::tiling2d_cube::{
        base_tests, compute_loop_tests, load_shared_memory_tests, outer_product_tests,
        write_output_tests,
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
    pub fn tiling2d_matmul_outer_product_vectorized_test_2() {
        outer_product_tests::tile_outer_product_vectorized_unit_test_2::<TestRuntime>(
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
    pub fn compute_loop_unit_offset_test() {
        compute_loop_tests::compute_loop_unit_offset_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn tiling2d_matmul_read_whole_vectorized_like_tile_test() {
        load_shared_memory_tests::read_whole_vectorized_like_tile_test::<TestRuntime>(
            &Default::default(),
        )
    }

    #[test]
    pub fn tiling2d_matmul_read_whole_vectorized_less_than_tile_test() {
        load_shared_memory_tests::read_whole_vectorized_less_than_tile_test::<TestRuntime>(
            &Default::default(),
        )
    }

    #[test]
    pub fn tiling2d_matmul_read_whole_scalar_test() {
        load_shared_memory_tests::read_whole_scalar_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn read_whole_scalar_out_of_bound_test() {
        load_shared_memory_tests::read_whole_scalar_out_of_bound_test::<TestRuntime>(
            &Default::default(),
        )
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
    pub fn load_lhs_transposed_cube_test() {
        load_shared_memory_tests::load_lhs_transposed_cube_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn load_lhs_transposed_out_of_bounds_cube_test() {
        load_shared_memory_tests::load_lhs_transposed_out_of_bounds_cube_test::<TestRuntime>(
            &Default::default(),
        )
    }

    #[test]
    pub fn load_lhs_transposed_offset_cube_test() {
        load_shared_memory_tests::load_lhs_transposed_offset_cube_test::<TestRuntime>(
            &Default::default(),
        )
    }

    #[test]
    pub fn load_rhs_plain_unit_test() {
        load_shared_memory_tests::load_rhs_plain_unit_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn load_rhs_plain_cube_test() {
        load_shared_memory_tests::load_rhs_plain_cube_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn load_rhs_plain_cube_offset_test() {
        load_shared_memory_tests::load_rhs_plain_cube_offset_test::<TestRuntime>(&Default::default())
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

    #[test]
    pub fn write_to_output_vectorized_less_than_tile_unit_test() {
        write_output_tests::write_to_output_vectorized_less_than_tile_unit_test::<TestRuntime>(
            &Default::default(),
        )
    }

    #[test]
    pub fn write_to_output_scalar_unit_test() {
        write_output_tests::write_to_output_scalar_unit_test::<TestRuntime>(&Default::default())
    }

    #[test]
    pub fn calculate_offsets_unit_test() {
        base_tests::calculate_offsets_unit_test::<TestRuntime>(&Default::default())
    }
}
