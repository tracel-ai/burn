mod base;
mod compute_loop;
mod config;
mod load_shared_memory;
mod outer_product;
mod tiling2d_core;
mod write_output;

pub use base::matmul_tiling_2d_cube;
pub use base::tests as base_tests;
pub use compute_loop::tests as compute_loop_tests;
pub use load_shared_memory::tests as load_shared_memory_tests;
pub use outer_product::tests as outer_product_tests;
pub use write_output::tests as write_output_tests;
