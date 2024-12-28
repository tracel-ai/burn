use std::marker::PhantomData;

use cubecl::{
    linalg::matmul::{
        components::{
            stage::{self, StageMatmulFamily},
            tile::{accelerated::Accelerated, TileMatmulFamily},
            MatmulSelection,
        },
        kernels::{matmul::AdvancedConfig, MatmulAvailabilityError},
    },
    prelude::*,
};

use super::{
    base::{ConvolutionConfigFactory, ConvolutionFamily, ConvolutionProblem},
    homogeneous::base::ImplicitGemmConvolutionFamily,
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
    ) -> <Self::GlobalConvolution as ConvolutionConfigFactory>::Config {
        let config = Self::GlobalConvolution::make_config(
            input,
            problem,
            cube_dim,
            cube_count,
            advanced_config,
        );
        problem.check_config(&config)?;
        Self::GlobalConvolution::check_config(&config)?;
        Ok(config)
    }

    /// Check availability of the matmul algorithm
    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        config: &<Self::GlobalConvolution as ConvolutionConfigFactory>::Config,
    ) -> Result<(), MatmulAvailabilityError> {
        Self::GlobalConvolution::check_availability::<R>(client, config)
    }

    /// Determine whether the given convolution problem is valid to launch (within hardware limits)
    fn can_launch<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &ConvolutionProblem,
        config: &<Self::GlobalConvolution as ConvolutionConfigFactory>::Config,
        selection: &Self::Selection,
    ) -> bool {
        if problem.options.groups > 1 || Self::check_availability::<R>(client, config).is_err() {
            return false;
        }

        let cube_count = Self::cube_count(selection, problem);
        let (max_x, max_y, max_z) = R::max_cube_count();
        match cube_count {
            CubeCount::Static(x, y, z) => x <= max_x && y <= max_y && z <= max_z,
            _ => true,
        }
    }
}

/// Cmma convolution
pub struct ImplicitCmmaConv;

impl Algorithm for ImplicitCmmaConv {
    type TileMatmul = Accelerated;
    type StageMatmul = stage::multi_buffer::MultiBufferMatmulFamily<Self::TileMatmul>;
    type GlobalConvolution = ImplicitGemmConvolutionFamily<Self::StageMatmul>;
    type Selection = MatmulSelection;
    type Input = <Self::GlobalConvolution as ConvolutionConfigFactory>::Input;

    fn cube_dim(selection: &MatmulSelection) -> CubeDim {
        CubeDim::new(selection.plane_dim, selection.num_stagess.m, 1)
    }

    fn cube_count(selection: &MatmulSelection, problem: &ConvolutionProblem) -> CubeCount {
        let m_stage = selection.num_stagess.m * selection.tile.m;
        let n_stage = selection.num_stagess.n * selection.tile.n;
        let cubes_needed_m = (problem.m as u32).div_ceil(m_stage);
        let cubes_needed_n = (problem.n as u32).div_ceil(n_stage);

        CubeCount::Static(cubes_needed_m, cubes_needed_n, 1)
    }
}
