mod base;
mod block_loop;
mod compute_loop;
mod config;
mod load_shared_memory;
mod outer_product;
#[cfg(feature = "export_tests")]
mod test_utils;
mod write_output;

pub use base::matmul_tiling_2d_cube;

#[cfg(feature = "export_tests")]
pub use {
    compute_loop::tests as compute_loop_tests,
    load_shared_memory::tests as load_shared_memory_tests,
    outer_product::tests as outer_product_tests, write_output::tests as write_output_tests,
};
