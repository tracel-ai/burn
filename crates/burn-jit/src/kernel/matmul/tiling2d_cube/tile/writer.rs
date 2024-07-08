use std::marker::PhantomData;

use burn_cube::prelude::*;

use crate::kernel::matmul::{
    config::CubeTiling2dConfig,
    tiling2d_cube::{
        base::Dimensions,
        write_output::{OutputWriter, WriteTileInfo},
    },
};

use super::{
    block_check::base::BlockCheck,
    loader::{CheckBounds, CheckBoundsExpand},
    memory_access::{MatchingVectorization, UnmatchingVectorization},
};
pub(crate) struct TileWriter<F: Float> {
    _f: PhantomData<F>,
}

#[cube]
impl<F: Float> OutputWriter<F> for TileWriter<F> {
    fn write_output<B: BlockCheck<F>>(
        out: &mut Tensor<F>,
        results: &Array<F>,
        write_info: WriteTileInfo,
        dims: Dimensions,
        config: Comptime<CubeTiling2dConfig>,
    ) {
        let vectorization = Comptime::vectorization(out);
        // let tile_size = Comptime::map(config, |c| c.tile_size);
        let match_tile = Comptime::map(vectorization, |v| v.val == 4); // TODO HARDCODED TO 4
        let coordinates = write_info.coordinates;

        let check_bounds = CheckBounds {
            dim_vertical: dims.m,
            dim_horizontal: dims.n,
            skip_row: coordinates.skip_row,
            skip_col: coordinates.skip_col,
        };

        if Comptime::get(match_tile) {
            B::write_output::<MatchingVectorization>(
                out,
                results,
                write_info,
                config,
                check_bounds,
            );
        } else {
            B::write_output::<UnmatchingVectorization>(
                out,
                results,
                write_info,
                config,
                check_bounds,
            );
        }
    }
}
