use cubecl::{
    linalg::matmul::components::{
        global::{
            self,
            homogeneous::{CyclicLoading, RhsLoader},
        },
        stage::{self, multi_buffer::RhsReader},
        Ident,
    },
    prelude::*,
};

use crate::kernel::conv::homogeneous::base::config;

#[cube]
/// Input to the convolution, responsible of filling the stage and providing a reader for it.
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
