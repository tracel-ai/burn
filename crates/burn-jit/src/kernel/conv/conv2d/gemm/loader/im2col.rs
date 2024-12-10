use cubecl::{
    linalg::matmul::components::{
        global::{
            args::{TensorArgs, TensorInput},
            Loader,
        },
        stage::{
            multi_buffer::LhsReader, ColMajorTiling, RowMajorTiling, Stage, TilingOrder as _,
            TilingOrderConfig,
        },
        Ident,
    },
    prelude::*,
};
use std::marker::PhantomData;

use crate::kernel::conv::{reader::im2col::Im2colReader, Config};

/// Loader that translates matrix coordinates to input coordinates using the `im2col` algorithm
#[derive(CubeType)]
pub struct SimpleIm2colLoader<EG: Numeric, ES: Numeric, G: Config> {
    pub tensor_view: Im2colReader<EG>,
    pub stage: Stage<ES>,
    _config: PhantomData<G>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: Config> Loader<EG, ES, G> for SimpleIm2colLoader<EG, ES, G> {
    type StageReader = LhsReader<ES>;

    fn fill_stage(this: &mut Self, #[comptime] config: G) {
        SimpleIm2col::load_to_slice::<EG, ES, G>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
            Ident::Lhs,
            config,
        );
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset);
    }

    fn as_stage_reader(this: &Self) -> Self::StageReader {
        LhsReader::new(this.stage)
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: Config> SimpleIm2colLoader<EG, ES, G> {
    pub fn new(
        tensor: TensorInput<EG, TensorArgs>,
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

        let tensor_view = Im2colReader::<EG> {
            tensor,
            m_offset: x_offset,
            k_offset: y_offset,
            stride_batch: tensor.stride(0),
            stride_y: tensor.stride(1),
            stride_x: tensor.stride(2),
            stride_channel: tensor.stride(3),
            shape_y: tensor.shape(1),
            shape_x: tensor.shape(2),
            shape_channel,
            shape_out_y,
            shape_out_x,

            shape_m,
            shape_k,
        };

        SimpleIm2colLoader::<EG, ES, G> {
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
    pub fn load_to_slice<EG: Numeric, ES: Numeric, G: Config>(
        read_view: &Im2colReader<EG>,
        slice: &mut SliceMut<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let stage_dim = config.stage_dim(ident);
        let line_size = config.global_line_size(ident);

        let num_stage_elements = stage_dim.total_elements();
        let total_units = comptime!(config.num_planes() * config.plane_dim());
        let jump_length = comptime!(total_units * line_size);
        let num_loads_per_unit = num_stage_elements / jump_length;

        #[allow(clippy::all)]
        let _ = comptime!(check_jump_divides_well(num_stage_elements, jump_length));

        let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        for i in 0..num_loads_per_unit {
            let unit_position = unit_position_base + i * jump_length;

            let tile_num_elements = stage_dim.tile_num_elements();
            let nth_tile = unit_position / tile_num_elements;
            let pos_within_tile = unit_position % tile_num_elements;

            let (tile_x, tile_y) = match config.tiling_order(ident) {
                TilingOrderConfig::RowMajor => RowMajorTiling::to_x_y(
                    nth_tile,
                    stage_dim.num_tiles_x_dim(),
                    stage_dim.num_tiles_y_dim(),
                ),
                TilingOrderConfig::ColMajor => ColMajorTiling::to_x_y(
                    nth_tile,
                    stage_dim.num_tiles_x_dim(),
                    stage_dim.num_tiles_y_dim(),
                ),
            };

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
