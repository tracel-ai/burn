use burn_cube::prelude::*;

use crate::kernel::matmul::{config::CubeTiling2dConfig, tiling2d_cube::base::Coordinates};

#[cube]
pub(crate) trait OutputWriter<F: Float>: Sync + Send + 'static {
    fn write_output(
        out: &mut Tensor<F>,
        results: &Array<F>,
        coordinates: Coordinates,
        offset_output: UInt,
        out_stride: UInt,
        config: Comptime<CubeTiling2dConfig>,
    );
}
