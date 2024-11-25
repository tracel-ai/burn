use cubecl::{
    linalg::matmul::components::{
        stage::{
            multi_buffer::RhsReader, ColMajorTiling, RowMajorTiling, Stage, TilingOrder,
            TilingOrderConfig,
        },
        Ident,
    },
    prelude::*,
};

use crate::kernel::conv::{
    homogeneous::loader::{check_jump_divides_well, Loader},
    Config,
};

#[derive(CubeType)]
pub struct WeightLoader<EG: Numeric, ES: Numeric> {
    pub tensor_view: WeightReader<EG>,
    pub stage: Stage<ES>,
}

#[cube]
impl<EG: Numeric, ES: Numeric, G: Config> Loader<EG, ES, G> for WeightLoader<EG, ES> {
    type StageReader = RhsReader<ES>;

    fn fill_stage(
        this: &mut Self,
        _test: &mut Tensor<EG>,
        #[comptime] config: G,
    ) -> Self::StageReader {
        WeightLoading::load_to_slice::<EG, ES, G>(
            &this.tensor_view,
            &mut this.stage.as_slice_mut(),
            Ident::Lhs,
            config,
        );
        RhsReader::new(this.stage)
    }

    fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, Ident::Rhs);
    }
}

#[cube]
impl<EG: Numeric, ES: Numeric> WeightLoader<EG, ES> {
    pub fn new<G: Config>(
        tensor: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        #[comptime] config: G,
    ) -> Self {
        let stage = Stage::new::<G::SmmConfig>(Ident::Rhs, config.to_smm_config());
        let tensor_view = WeightReader::new(tensor, x_offset, y_offset);

        WeightLoader::<EG, ES> { tensor_view, stage }
    }
}

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct WeightLoading;

#[cube]
impl WeightLoading {
    pub fn load_to_slice<EG: Numeric, ES: Numeric, G: Config>(
        read_view: &WeightReader<EG>,
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
                read_view.load_coalesced::<G>(tile_x, tile_y, pos_within_tile, ident, config);

            slice[unit_position / line_size] = Line::cast_from(line_read);
        }
    }
}

#[derive(CubeType)]
/// A view of a tensor that starts reading data from a specified offset.
/// Ensures safe access by preventing out-of-bounds errors.
/// Includes pre-fetched shapes and strides for optimized performance.
pub struct WeightReader<E: Numeric> {
    pub tensor: *const Tensor<Line<E>>,
    pub x_offset: u32,
    pub y_offset: u32,
    pub stride_k_y: u32,
    pub stride_k_x: u32,
    pub stride_in_c: u32,
    pub stride_out_c: u32,
    pub shape_k_y: u32,
    pub shape_k_x: u32,
    pub shape_in_c: u32,
    pub shape_out_c: u32,
}

unsafe impl<E: Numeric> Sync for WeightReader<E> {}
unsafe impl<E: Numeric> Send for WeightReader<E> {}

#[cube]
impl<EG: Numeric> WeightReader<EG> {
    /// Instantiate a read view over the given tensor, pre-fetching needed strides and shapes
    pub fn new(tensor: &Tensor<Line<EG>>, x_offset: u32, y_offset: u32) -> Self {
        let stride_k_y = tensor.stride(0);
        let stride_k_x = tensor.stride(1);
        let stride_in_c = tensor.stride(2);
        let stride_out_c = tensor.stride(3);

        let shape_k_y = tensor.shape(0);
        let shape_k_x = tensor.shape(1);
        let shape_in_c = tensor.shape(2);
        let shape_out_c = tensor.shape(3);

        WeightReader::<EG> {
            tensor,
            x_offset,
            y_offset,
            stride_k_y,
            stride_k_x,
            stride_in_c,
            stride_out_c,
            shape_k_y,
            shape_k_x,
            shape_in_c,
            shape_out_c,
        }
    }

    /// Advance the view along the k dimension by a specified offset, `k_offset`.
    pub fn update_view(&mut self, k_offset: u32, #[comptime] ident: Ident) {
        match ident {
            Ident::Lhs => {
                self.y_offset += k_offset;
            }
            Ident::Rhs => {
                self.x_offset += k_offset;
            }
            Ident::Out => {}
        }
    }

    /// Reads data from the tensor view at the specified tile coordinates (tile_x, tile_y).
    ///
    /// Each unit loads one line in a coalesced manner for improved efficiency.
    /// For row-major tensors, subsequent units read lines horizontally within the tile,
    /// while for column-major tensors, they read lines vertically.
    ///
    /// # Note
    ///
    /// Out-of-bounds reads will be translated to zeros.
    pub fn load_coalesced<G: Config>(
        &self,
        tile_x: u32,
        tile_y: u32,
        unit_id: u32,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) -> Line<EG> {
        let line_size = config.global_line_size(ident);
        let tile_size_x = config.stage_dim(ident).tile_size_x_dim();
        let tile_size_y = config.stage_dim(ident).tile_size_y_dim();

        let view_tile_x = tile_x * tile_size_x + self.x_offset;
        let view_tile_y = tile_y * tile_size_y + self.y_offset;

        let (load_x, load_y) = (unit_id / tile_size_y, unit_id % tile_size_y);

        let view_x = view_tile_x + load_x;
        let view_y = view_tile_y + load_y;

        let in_c = view_x % self.shape_in_c;
        let rem = view_x / self.shape_in_c;
        let k_x = rem % self.shape_k_x;
        let k_y = rem / self.shape_k_x;

        let read_pos =
            k_y * self.stride_k_y + k_x * self.stride_k_x + in_c * self.stride_in_c + view_y;

        let read_pos = read_pos / line_size;

        let in_bounds_k = comptime!(!config.check_k_bounds()) || k_y < self.shape_k_y;
        let in_bounds_n = comptime!(!config.check_n_bounds()) || view_y < self.shape_out_c;

        select(
            in_bounds_k && in_bounds_n,
            self.read(read_pos),
            Line::empty(line_size).fill(EG::from_int(0)),
        )
    }

    fn read(&self, position: u32) -> Line<EG> {
        unsafe { *(*self.tensor).index_unchecked(position) }
    }
}
