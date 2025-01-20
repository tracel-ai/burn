use cubecl::{
    linalg::matmul::{
        components::{
            stage::{self, StageMatmulFamily},
            tile::{accelerated::Accelerated, TileMatmulFamily},
            InvalidConfigError,
        },
        kernels::matmul::AdvancedConfig,
    },
    prelude::*,
};

use super::{
    base::{ConvolutionConfigFactory, ConvolutionFamily, ConvolutionProblem},
    homogeneous::base::ImplicitGemmConvolutionFamily,
    selection::ConvSelection,
};

/// Specifications for a convolution algorithm
pub trait Algorithm {
    type TileMatmul: TileMatmulFamily;
    type StageMatmul: StageMatmulFamily;
    type GlobalConvolution: ConvolutionFamily<Self::StageMatmul, Input = Self::Input>;
    type Selection;
    type Input;

    fn cube_dim(selection: &Self::Selection) -> CubeDim;
    fn cube_count(selection: &Self::Selection, problem: &ConvolutionProblem) -> CubeCount;

    /// Make a convolution config from a convolution problem, and launch options
    fn make_config(
        input: <Self::GlobalConvolution as ConvolutionConfigFactory>::Input,
        problem: &ConvolutionProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Result<<Self::GlobalConvolution as ConvolutionConfigFactory>::Config, InvalidConfigError>
    {
        let config = Self::GlobalConvolution::make_config(
            input,
            problem,
            cube_dim,
            cube_count,
            advanced_config,
        );
        Self::GlobalConvolution::check_config(&config)?;
        Ok(config)
    }
}

/// Cmma convolution
pub struct ImplicitCmmaConv;

impl Algorithm for ImplicitCmmaConv {
    type TileMatmul = Accelerated;
    type StageMatmul = stage::multi_buffer::MultiBufferMatmulFamily<Self::TileMatmul>;
    type GlobalConvolution = ImplicitGemmConvolutionFamily<Self::StageMatmul>;
    type Selection = ConvSelection;
    type Input = <Self::GlobalConvolution as ConvolutionConfigFactory>::Input;

    fn cube_dim(selection: &ConvSelection) -> CubeDim {
        CubeDim::new(
            selection.matmul.plane_dim,
            selection.matmul.num_stagess.m,
            1,
        )
    }

    fn cube_count(selection: &ConvSelection, problem: &ConvolutionProblem) -> CubeCount {
        let m_stage = selection.matmul.num_stagess.m * selection.matmul.tile.m;
        let n_stage = selection.matmul.num_stagess.n * selection.matmul.tile.n;
        let cubes_needed_m = (problem.m as u32).div_ceil(m_stage);
        let cubes_needed_n = (problem.n as u32).div_ceil(n_stage);

        CubeCount::Static(cubes_needed_m, cubes_needed_n, 1)
    }
}
