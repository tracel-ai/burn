mod base;
mod block_loop;
mod compute_loop;
mod direct;
mod launch;
mod load_shared_memory;
mod outer_product;
#[cfg(feature = "export_tests")]
mod test_utils;
mod tile;
mod write_output;

pub use launch::matmul_tiling_2d_cube;

#[cfg(feature = "export_tests")]
pub use {
    compute_loop::tests as compute_loop_tests, outer_product::tests as outer_product_tests,
    tile::tile_loading::tests as load_shared_memory_tests,
    tile::tile_read::tests as tile_read_tests, tile::tile_write::tests as tile_write_tests,
    write_output::tests as write_output_tests,
};
