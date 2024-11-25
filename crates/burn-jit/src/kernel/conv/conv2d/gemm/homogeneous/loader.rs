use std::marker::PhantomData;

use cubecl::{
    linalg::matmul::components::{
        global::{
            self,
            homogeneous::{CyclicLoading, RhsLoader},
            AccumulatorLoader,
        },
        stage::{
            self,
            multi_buffer::{LhsReader, RhsReader},
            Config as _, Stage,
        },
        tile, Ident,
    },
    prelude::*,
};

use crate::kernel::conv::conv2d::gemm::{input_reader::Im2colReader, Config};

use super::base::config;

#[cube]
/// Input to the global matmul, responsible of filling the stage and providing a reader for it.
/// Advances along the k-dimension to fill the stage with further data.
pub trait Loader<EG: Numeric, ES: Numeric, G: global::Config>:
    CubeType + 'static + Send + Sync
{
    /// The stage reader which matches the input of the underlying stage matmul.
    type StageReader: CubeType;

    /// Fills the stage at the current k offset and returns a reader for it.
    fn fill_stage(this: &mut Self, #[comptime] config: G) -> Self::StageReader;

    /// Move the k offset by k_offset
    fn advance_view(this: &mut Self, k_offset: u32);
}

#[cube]
impl<EG: Numeric, ES: Numeric, S: stage::Config> Loader<EG, ES, config::Config<S>>
    for RhsLoader<EG, ES, S>
{
    type StageReader = RhsReader<ES>;

    fn fill_stage(this: &mut Self, #[comptime] config: config::Config<S>) -> Self::StageReader {
        CyclicLoading::load_to_slice::<EG, ES, config::Config<S>>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
            Ident::Rhs,
            config,
        );
        RhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}

#[derive(CubeType)]
pub struct SimpleIm2colLoader<EG: Numeric, ES: Numeric, G: Config> {
    pub tensor_view: Im2colReader<EG>,
    pub stage: Stage<ES>,
    _config: PhantomData<G>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: Config> Loader<EG, ES, G> for SimpleIm2colLoader<EG, ES, G> {
    type StageReader = LhsReader<ES>;

    fn fill_stage(this: &mut Self, #[comptime] config: G) -> Self::StageReader {
        im2col::SimpleIm2col::load_to_slice::<EG, ES, G>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
            Ident::Lhs,
            config,
        );
        LhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: Config> SimpleIm2colLoader<EG, ES, G> {
    pub fn new(
        tensor: &Tensor<Line<EG>>,
        shape_out_y: u32,
        shape_out_x: u32,
        x_offset: u32,
        y_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Lhs, config.to_smm_config());
        let tensor_view = Im2colReader::<EG> {
            tensor,
            m_offset: x_offset,
            k_offset: y_offset,
            stride_batch: tensor.stride(0),
            stride_y: tensor.stride(1),
            stride_x: tensor.stride(2),
            stride_channel: tensor.stride(3),
            shape_batch: tensor.shape(0),
            shape_y: tensor.shape(1),
            shape_x: tensor.shape(2),
            shape_channel: tensor.shape(3),
            shape_out_y,
            shape_out_x,
        };

        SimpleIm2colLoader::<EG, ES, G> {
            tensor_view,
            stage,
            _config: PhantomData::<G>.runtime(),
        }
    }
}

#[derive(CubeType)]
/// A view of a tensor that starts reading data from a specified offset.
/// Ensures safe access by preventing out-of-bounds errors.
/// Includes pre-fetched shapes and strides for optimized performance.
pub struct BiasReader<E: Numeric> {
    pub tensor: *const Tensor<Line<E>>,
    pub n_offset: u32,
    pub shape_n: u32,
}

unsafe impl<E: Numeric> Sync for BiasReader<E> {}
unsafe impl<E: Numeric> Send for BiasReader<E> {}

#[cube]
impl<E: Numeric> BiasReader<E> {
    pub fn load_simple<G: stage::Config>(
        &self,
        tile_n: u32,
        unit_id: u32,
        #[comptime] config: G,
    ) -> Line<E> {
        let line_size = config.line_size(Ident::Out);
        let tile_size = config.stage_dim(Ident::Rhs).tile_size_y_dim();

        let view_tile_n = tile_n * tile_size + self.n_offset;

        let view_n = view_tile_n + unit_id;
        let read_pos = view_n / line_size;

        select(
            view_n < self.shape_n,
            self.read(read_pos),
            Line::empty(line_size).fill(E::from_int(0)),
        )
    }

    fn read(&self, position: u32) -> Line<E> {
        unsafe { *(*self.tensor).index_unchecked(position) }
    }
}

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

            let num_stage_elements = stage_dim.height();
            let tile_num_elements = stage_dim.tile_size_y_dim();

            let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
            let unit_position_base = unit_id * line_size;

            let mut slice = this.stage.as_slice_mut();

            if unit_position_base < num_stage_elements {
                let nth_tile = unit_position_base / tile_num_elements;
                slice[unit_id] = Line::<Acc>::cast_from(this.tensor_view.load_simple::<G>(
                    nth_tile,
                    unit_position_base,
                    config,
                ));
            }
        }
        LhsReader::new(this.stage)
    }

    /// Load accumulator
    fn load<I: Numeric, Tile: tile::Matmul<I, Acc>>(
        this: &mut Self,
        acc: &mut Tile::Accumulator,
        n_tile: u32,
        #[comptime] config: Tile::Config,
    ) {
        if this.has_bias {
            let start = n_tile * Tile::N;
            let slice = this.stage.as_slice_mut().slice(start, start + Tile::N);
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
                    comptime!(config.stage_dim(Ident::Rhs).height() / line_size),
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

pub fn check_jump_divides_well(num_stage_elements: u32, jump_length: u32) {
    assert!(
        num_stage_elements % jump_length == 0,
        "Too many data will be loaded, resulting in out of bounds. 
    Try setting line size and number of planes so that jump_length divides num_stage_elements."
    );
}

mod im2col {
    use cubecl::linalg::matmul::components::stage::{
        ColMajorTiling, RowMajorTiling, TilingOrder, TilingOrderConfig,
    };

    use super::*;

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
}
