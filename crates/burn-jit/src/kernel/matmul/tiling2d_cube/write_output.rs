use burn_cube::prelude::*;

use crate::kernel::matmul::config::CubeTiling2dConfig;

use super::{
    base::{Coordinates, Dimensions},
    direct::block_check::{
        base::BlockCheck, horizontal_block_check::HorizontalBlockCheck,
        unchecked_block::UncheckedBlockCheck, vertical_block_check::VerticalBlockCheck,
        whole_block_check::WholeBlockCheck,
    },
};

#[cube]
pub(crate) trait OutputWriter<F: Float>: Sync + Send + 'static {
    fn write_output<B: BlockCheck<F>>(
        out: &mut Tensor<F>,
        results: &Array<F>,
        coordinates: Coordinates,
        offset_output: UInt,
        out_stride: UInt,
        dims: Dimensions,
        config: Comptime<CubeTiling2dConfig>,
    );
}

#[cube]
pub(crate) fn write_to_output<F: Float, W: OutputWriter<F>>(
    out: &mut Tensor<F>,
    results: &Array<F>,
    coordinates: Coordinates,
    offset_output: UInt,
    dims: Dimensions,
    config: Comptime<CubeTiling2dConfig>,
) {
    let check_m_bounds = Comptime::map(config, |c| c.check_m_bounds);
    let check_n_bounds = Comptime::map(config, |c| c.check_n_bounds);

    let out_stride = dims.n;

    if Comptime::get(check_m_bounds) {
        if Comptime::get(check_n_bounds) {
            W::write_output::<WholeBlockCheck>(
                out,
                results,
                coordinates,
                offset_output,
                out_stride,
                dims,
                config,
            );
        } else {
            W::write_output::<VerticalBlockCheck>(
                out,
                results,
                coordinates,
                offset_output,
                out_stride,
                dims,
                config,
            );
        }
    } else {
        if Comptime::get(check_n_bounds) {
            W::write_output::<HorizontalBlockCheck>(
                out,
                results,
                coordinates,
                offset_output,
                out_stride,
                dims,
                config,
            );
        } else {
            W::write_output::<UncheckedBlockCheck>(
                out,
                results,
                coordinates,
                offset_output,
                out_stride,
                dims,
                config,
            );
        }
    }
}
