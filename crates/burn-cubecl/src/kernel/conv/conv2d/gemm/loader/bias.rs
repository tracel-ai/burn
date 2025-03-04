use crate::kernel::conv::{
    homogeneous::base::ConvTilingLayout, precision::ConvPrecision, reader::bias::BiasReader,
};
use cubecl::{
    linalg::matmul::components::{
        global::AccumulatorLoader,
        stage::{Stage, StageConfig},
        tile::{Tile, TileConfig, TileMatmul},
        Ident,
    },
    prelude::*,
};
use cubecl_std::tensor::r#virtual::VirtualTensor;
use std::marker::PhantomData;

/// Special loader to broadcast the 1D bias to the 2D accumulator matrix
#[derive(CubeType)]
pub struct BiasLoader<CS: ConvPrecision, G: StageConfig> {
    pub tensor_view: BiasReader<CS::EG>,
    pub stage: Stage<CS::EA, ConvTilingLayout>,
    pub has_bias: bool,
    _config: PhantomData<G>,
}

#[cube]
impl<CS: ConvPrecision, G: StageConfig> AccumulatorLoader<CS::EG, CS::EA, G> for BiasLoader<CS, G> {
    fn fill_stage(this: &mut Self, #[comptime] config: G) {
        if this.has_bias {
            let stage_tiling = config.tiling_dimensions(Ident::Rhs);
            let line_size = config.line_size(Ident::Out);

            let num_stage_elements = stage_tiling.total_col();

            let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
            let unit_position_base = unit_id * line_size;

            let mut slice = this.stage.as_slice_mut();

            if unit_position_base < num_stage_elements {
                let read_line = this
                    .tensor_view
                    .load_simple::<G>(unit_position_base, config);
                slice[unit_id] = Line::cast_from(read_line);
            }
        }
    }

    /// Load accumulator
    fn load<I: Numeric, TMM: TileMatmul<I, CS::EA>>(
        this: &mut Self,
        acc: &mut TMM::Accumulator,
        tile_n: u32,
        #[comptime] config: TMM::Config,
    ) {
        if this.has_bias {
            let line_size = config.line_size(Ident::Out);
            let tile_elems = config.tile_shape().n / line_size;
            let start = tile_n * tile_elems;
            let slice = this.stage.as_slice_mut().slice(start, start + tile_elems);
            let tile = Tile::new_strided(slice, 0);
            TMM::fill_accumulator(&tile, acc, config);
        } else {
            TMM::zero_accumulator(acc, config);
        }
    }
}

#[cube]
impl<CS: ConvPrecision, G: StageConfig> BiasLoader<CS, G> {
    pub fn new(
        tensor: VirtualTensor<CS::EG>,
        n_offset: u32,
        #[comptime] config: G,
        #[comptime] has_bias: bool,
    ) -> Self {
        if has_bias {
            let stage = init_stage::<CS::EA, G>(config);
            let shape_n = tensor.shape(0);
            let tensor_view = BiasReader::<CS::EG>::new(tensor, n_offset, shape_n);

            BiasLoader::<CS, G> {
                tensor_view,
                stage,
                has_bias,
                _config: PhantomData::<G>.runtime(),
            }
        } else {
            let stage = init_empty_stage::<CS::EA>();
            let tensor_view = BiasReader::<CS::EG>::new(tensor, 0, 0);
            BiasLoader::<CS, G> {
                stage,
                tensor_view,
                has_bias,
                _config: PhantomData::<G>.runtime(),
            }
        }
    }
}

#[cube]
fn init_stage<ES: Numeric, G: StageConfig>(#[comptime] config: G) -> Stage<ES, ConvTilingLayout> {
    let line_size = config.line_size(Ident::Out);

    let smem = SharedMemory::new_lined(
        comptime!(config.tiling_dimensions(Ident::Rhs).total_col() / line_size),
        line_size,
    );

    Stage::<ES, ConvTilingLayout>::new_with_smem(smem)
}

#[cube]
fn init_empty_stage<ES: Numeric>() -> Stage<ES, ConvTilingLayout> {
    Stage::<ES, ConvTilingLayout>::new_with_smem(SharedMemory::new(1))
}
