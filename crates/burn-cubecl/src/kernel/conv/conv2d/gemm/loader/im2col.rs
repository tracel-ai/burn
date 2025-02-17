use cubecl::prelude::pipeline::Pipeline;
use cubecl::{
    linalg::{
        matmul::components::{
            global::InputLoader,
            stage::{multi_buffer::LhsReader, Stage, TilingLayout},
            Ident,
        },
        tensor::VirtualTensor,
    },
    prelude::*,
};
use std::marker::PhantomData;

use crate::kernel::conv::{precision::ConvPrecision, reader::im2col::Im2colReader, ConvGemmConfig};

/// Loader that translates matrix coordinates to input coordinates using the `im2col` algorithm
#[derive(CubeType)]
pub struct SimpleIm2colLoader<CS: ConvPrecision, G: ConvGemmConfig> {
    pub tensor_view: Im2colReader<CS::EG>,
    pub stage: Stage<CS::ES>,
    _config: PhantomData<G>,
}

#[cube]
impl<CS: ConvPrecision, G: ConvGemmConfig> InputLoader<CS::EG, CS::ES, G>
    for SimpleIm2colLoader<CS, G>
{
    type StageReader = LhsReader<CS::ES>;

    fn fill_stage(this: &mut Self, #[comptime] config: G) {
        SimpleIm2col::load_to_slice::<CS, G>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
            Ident::Lhs,
            config,
        );
    }

    /// Fills the stage at the current k offset.
    fn fill_stage_window(_this: &mut Self, _pipeline: Pipeline<CS::ES>, #[comptime] _config: G) {
        comptime!(todo!());
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset);
    }

    fn as_stage_reader(this: &Self) -> Self::StageReader {
        LhsReader::new(this.stage)
    }
}

#[cube]
impl<CS: ConvPrecision, G: ConvGemmConfig> SimpleIm2colLoader<CS, G> {
    pub fn new(
        tensor: VirtualTensor<CS::EG>,
        shape_out_y: u32,
        shape_out_x: u32,
        x_offset: u32,
        y_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
        let shape_batch = tensor.shape(0);
        let shape_channel = tensor.shape(3);

        let shape_m = shape_batch * shape_out_y * shape_out_x;
        let shape_k = shape_channel * config.kernel_size(0) * config.kernel_size(1);

        let tensor_view = Im2colReader::<CS::EG>::new(
            tensor,
            shape_out_y,
            shape_out_x,
            x_offset,
            y_offset,
            shape_k,
            shape_channel,
            shape_m,
        );

        SimpleIm2colLoader::<CS, G> {
            tensor_view,
            stage,
            _config: PhantomData::<G>.runtime(),
        }
    }
}

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct SimpleIm2col;

#[cube]
impl SimpleIm2col {
    pub fn load_to_slice<CS: ConvPrecision, G: ConvGemmConfig>(
        read_view: &Im2colReader<CS::EG>,
        slice: &mut SliceMut<Line<CS::ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let stage_tiling = config.stage_tiling(ident);
        let line_size = config.global_line_size(ident);

        let num_stage_elements = stage_tiling.total_size();
        let total_units = comptime!(config.num_planes() * config.plane_dim());
        let jump_length = comptime!(total_units * line_size);
        let num_loads_per_unit = num_stage_elements / jump_length;

        #[allow(clippy::all)]
        let _ = comptime!(check_jump_divides_well(num_stage_elements, jump_length));

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        for i in 0..num_loads_per_unit {
            let unit_position = unit_position_base + i * jump_length;

            let tile_num_elements = stage_tiling.tile_size();
            let nth_tile = unit_position / tile_num_elements;
            let pos_within_tile = unit_position % tile_num_elements;

            let (tile_x, tile_y) = TilingLayout::to_x_y(
                config.tiling_layout(ident),
                nth_tile,
                stage_tiling.tile_count_row(),
                stage_tiling.tile_count_col(),
            );

            let line_read =
                read_view.load_simple::<G>(tile_x, tile_y, pos_within_tile, ident, config);

            slice[unit_position / line_size] = Line::cast_from(line_read);
        }
    }
}

pub fn check_jump_divides_well(num_stage_elements: u32, jump_length: u32) {
    assert!(
        num_stage_elements % jump_length == 0,
        "Too many data will be loaded, resulting in out of bounds. 
    Try setting line size and number of planes so that jump_length divides num_stage_elements."
    );
}
