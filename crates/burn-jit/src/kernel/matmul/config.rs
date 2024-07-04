use burn_cube::{
    compute::CubeCount,
    frontend::{CubeContext, Init, UInt},
    ir::CubeDim,
};
use burn_tensor::Shape;

#[derive(Debug, Clone)]
/// Tiling 2D parameters
pub struct Tiling2dConfig {
    /// Block size along dimension of lhs
    pub block_size_m: usize,
    /// Block size along common dimension
    pub block_size_k: usize,
    /// Block size along dimension of rhs
    pub block_size_n: usize,
    /// Tile size and shared memory vectorization
    pub tile_size: usize,
    /// Loop unrolling
    pub unroll: bool,
}

impl Default for Tiling2dConfig {
    fn default() -> Self {
        Self {
            block_size_m: 64,
            block_size_k: 32,
            block_size_n: 64,
            tile_size: 4,
            unroll: false,
        }
    }
}

impl Init for CubeTiling2dConfig {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

#[derive(Debug, Clone, Copy)]
/// Tiling 2D parameters
pub struct CubeTiling2dConfig {
    /// Block size along dimension of lhs
    pub block_size_m: UInt,
    /// Block size along common dimension
    pub block_size_k: UInt,
    /// Block size along dimension of rhs
    pub block_size_n: UInt,
    /// Loop unrolling for inner compute loop. Probably slower
    pub unroll_compute: bool,
    /// Loop unrolling for all loops related to vectorization/tile size. Probably faster
    pub unroll_tile: bool,
    /// Bounds must be checked on lhs dimension
    pub check_m_bounds: bool,
    /// Bounds must be checked on common dimension
    pub check_k_bounds: bool,
    /// Bounds must be checked on rhs dimension
    pub check_n_bounds: bool,
    /// Bounds must be checked on shared memory write
    pub check_sm_bounds: bool,
    /// Tile size. Should correspond to vectorization of inputs/outputs/shared memory
    pub tile_size: UInt,
    /// Lhs is transposed in global memory
    pub lhs_transposed: bool,
    /// Rhs is transposed in global memory
    pub rhs_transposed: bool,
}

impl CubeTiling2dConfig {
    pub fn new(
        config: &Tiling2dConfig,
        m: usize,
        k: usize,
        n: usize,
        lhs_transposed: bool,
        rhs_transposed: bool,
    ) -> Self {
        assert!(
            config.block_size_k <= config.block_size_m
                && config.block_size_k <= config.block_size_n,
            "Larger block size in k than m or n results in unfilled shared memory."
        );
        assert!(
            config.block_size_m % config.tile_size == 0
                && config.block_size_k % config.tile_size == 0
                && config.block_size_n % config.tile_size == 0,
            "Tiling 2d algorithm assumes tile size divides block size perfectly. "
        );
        CubeTiling2dConfig {
            block_size_m: UInt::new(config.block_size_m as u32),
            block_size_k: UInt::new(config.block_size_k as u32),
            block_size_n: UInt::new(config.block_size_n as u32),
            unroll_compute: config.unroll,
            unroll_tile: true,
            check_m_bounds: m % config.block_size_m != 0,
            check_k_bounds: k % config.block_size_k != 0,
            check_n_bounds: n % config.block_size_n != 0,
            check_sm_bounds: config.block_size_k != config.block_size_m
                || config.block_size_k != config.block_size_n,
            tile_size: UInt::new(config.tile_size as u32),
            lhs_transposed,
            rhs_transposed,
        }
    }
}

pub fn tiling2d_cube_count<const D: usize>(
    output_shape: &Shape<D>,
    config: &Tiling2dConfig,
) -> CubeCount {
    let num_rows = output_shape.dims[D - 2];
    let num_cols = output_shape.dims[D - 1];

    // set number of workgroups
    let blocks_needed_in_x = f32::ceil(num_rows as f32 / config.block_size_m as f32) as u32;
    let blocks_needed_in_y = f32::ceil(num_cols as f32 / config.block_size_n as f32) as u32;
    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output_shape.dims[i];
    }

    CubeCount::new(blocks_needed_in_x, blocks_needed_in_y, num_iter as u32)
}

pub fn tiling2d_cube_dim(config: &Tiling2dConfig) -> CubeDim {
    CubeDim::new(
        (config.block_size_m / config.tile_size) as u32,
        (config.block_size_n / config.tile_size) as u32,
        1,
    )
}
