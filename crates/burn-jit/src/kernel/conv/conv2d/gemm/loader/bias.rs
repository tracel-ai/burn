use std::marker::PhantomData;

use cubecl::{
    linalg::matmul::components::{
        global::AccumulatorLoader,
        stage::{self, multi_buffer::LhsReader, Stage},
        tile::{self, Config as _},
        Ident,
    },
    prelude::*,
};

use crate::kernel::conv::reader::bias::BiasReader;

/// Special loader to broadcast the 1D bias to the 2D accumulator matrix
#[derive(CubeType)]
pub struct BiasLoader<O: Numeric, Acc: Numeric, G: stage::Config> {
    pub tensor_view: BiasReader<O>,
    pub stage: Stage<Acc>,
    pub has_bias: bool,
    _config: PhantomData<G>,
}

#[cube]
impl<O: Numeric, Acc: Numeric, G: stage::Config> AccumulatorLoader<O, Acc, G>
    for BiasLoader<O, Acc, G>
{
    type StageReader = LhsReader<Acc>;

    fn fill_stage(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
        if this.has_bias {
            let stage_dim = config.stage_dim(Ident::Rhs);
            let line_size = config.line_size(Ident::Out);

            let num_stage_elements = stage_dim.width();

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
        LhsReader::new(this.stage)
    }

    /// Load accumulator
    fn load<I: Numeric, Tile: tile::Matmul<I, Acc>>(
        this: &mut Self,
        acc: &mut Tile::Accumulator,
        tile_n: u32,
        #[comptime] config: Tile::Config,
    ) {
        if this.has_bias {
            let line_size = config.line_size(Ident::Out);
            let tile_elems = Tile::N / line_size;
            let start = tile_n * tile_elems;
            let slice = this.stage.as_slice_mut().slice(start, start + tile_elems);
            Tile::fill_accumulator(&slice, acc, 0, config);
        } else {
            Tile::zero_accumulator(acc, config);
        }
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: stage::Config> BiasLoader<EG, ES, G> {
    pub fn new(
        tensor: &Tensor<Line<EG>>,
        n_offset: u32,
        #[comptime] config: G,
        #[comptime] has_bias: bool,
    ) -> Self {
        if has_bias {
            let stage = {
                let line_size = config.line_size(Ident::Out);

                let smem = SharedMemory::new_lined(
                    comptime!(config.stage_dim(Ident::Rhs).width() / line_size),
                    line_size,
                );

                Stage::<ES> { smem }
            };
            let tensor_view = BiasReader::<EG> {
                tensor,
                n_offset,
                shape_n: tensor.shape(0),
            };

            BiasLoader::<EG, ES, G> {
                tensor_view,
                stage,
                has_bias,
                _config: PhantomData::<G>.runtime(),
            }
        } else {
            let stage = Stage::<ES> {
                smem: SharedMemory::new(1),
            };
            let tensor_view = BiasReader::<EG> {
                tensor,
                n_offset: 0,
                shape_n: 0,
            };
            BiasLoader::<EG, ES, G> {
                stage,
                tensor_view,
                has_bias,
                _config: PhantomData::<G>.runtime(),
            }
        }
    }
}
