use cubecl::{linalg::matmul::components::Ident, prelude::*};

use crate::kernel::conv::Config;

#[derive(CubeType)]
/// A view of a feature map tensor that starts reading data from a specified offset.
/// Ensures safe access by preventing out-of-bounds errors.
/// Includes pre-fetched shapes and strides for optimized performance.
pub struct Im2colReader<E: Numeric> {
    pub tensor: *const Tensor<Line<E>>,
    pub m_offset: u32,
    pub k_offset: u32,

    pub stride_batch: u32,
    pub stride_y: u32,
    pub stride_x: u32,
    pub stride_channel: u32,

    pub shape_y: u32,
    pub shape_x: u32,
    pub shape_channel: u32,

    pub shape_out_y: u32,
    pub shape_out_x: u32,

    pub shape_m: u32,
    pub shape_k: u32,
}

unsafe impl<E: Numeric> Sync for Im2colReader<E> {}
unsafe impl<E: Numeric> Send for Im2colReader<E> {}

#[cube]
impl<F: Numeric> Im2colReader<F> {
    /// Advance the view along the k dimension by a specified offset, `k_offset`.
    pub fn update_view(&mut self, k_offset: u32) {
        self.k_offset += k_offset;
    }

    /// Reads data from the tensor view at the specified tile coordinates (tile_x, tile_y) using
    /// the `im2col` algorithm to translate them to input coordinates.
    ///
    /// Each unit loads one line in a coalesced manner for improved efficiency.
    /// For row-major tensors, subsequent units read lines horizontally within the tile,
    /// while for column-major tensors, they read lines vertically.
    ///
    /// # Note
    ///
    /// Out-of-bounds reads will be translated to zeros.
    pub fn load_simple<G: Config>(
        &self,
        tile_x: u32,
        tile_y: u32,
        unit_id: u32,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) -> Line<F> {
        let line_size = config.global_line_size(ident);
        let tile_size_x = config.stage_dim(ident).tile_size_x_dim();
        let tile_size_y = config.stage_dim(ident).tile_size_y_dim();

        let view_tile_m = tile_x * tile_size_x + self.m_offset;
        let view_tile_k = tile_y * tile_size_y + self.k_offset;

        let load_m = unit_id / tile_size_y;
        let load_k = unit_id % tile_size_y;

        let view_m = view_tile_m + load_m;
        let view_k = view_tile_k + load_k;

        let out_x = view_m % self.shape_out_x;
        let rem = view_m / self.shape_out_x;
        let out_y = rem % self.shape_out_y;
        let batch = rem / self.shape_out_y;

        let kernel_w = config.kernel_size(1);

        let channel = view_k % self.shape_channel;
        let rem = view_k / self.shape_channel;
        let kernel_x = rem % kernel_w;
        let kernel_y = rem / kernel_w;

        let y =
            (out_y * config.stride(0) + kernel_y * config.dilation(0)) as i32 - config.padding(0);
        let x =
            (out_x * config.stride(1) + kernel_x * config.dilation(1)) as i32 - config.padding(1);

        let m_in_bounds = comptime!(!config.check_m_bounds()) || view_m < self.shape_m;
        let k_in_bounds = comptime!(!config.check_k_bounds()) || view_k < self.shape_k;
        let no_padding = comptime!(config.padding(0) == 0 && config.padding(1) == 0);
        let hw_in_bounds = no_padding
            || (y >= 0 && (y as u32) < self.shape_y && x >= 0 && (x as u32) < self.shape_x);
        let in_bounds = m_in_bounds && k_in_bounds && hw_in_bounds;
        let read_pos = batch * self.stride_batch
            + y as u32 * self.stride_y
            + x as u32 * self.stride_x
            + channel * self.stride_channel;

        let read_pos = read_pos / line_size;

        let mut res = Line::empty(line_size).fill(F::from_int(0));
        if in_bounds {
            res = self.read(read_pos);
        }

        res
    }

    fn read(&self, position: u32) -> Line<F> {
        unsafe { *(*self.tensor).index_unchecked(position) }
    }
}
