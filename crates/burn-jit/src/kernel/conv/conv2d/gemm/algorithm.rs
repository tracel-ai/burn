use std::marker::PhantomData;

use cubecl::{
    linalg::matmul::{
        components::{
            stage::{self, StageSize},
            tile::{self, accelerated::Accelerated16x16x16, Matmul as _},
            MatmulKernel,
        },
        kernels::{matmul::AdvancedConfig, MatmulAvailabilityError},
    },
    prelude::*,
};

use super::{
    base::{Convolution, ConvolutionKernel, ConvolutionLaunch, ConvolutionProblem},
    homogeneous::base::ImplicitGemmConvolution,
    spec::ConvSpec,
};

/// Specifications for a convolution algorithm
pub trait Algorithm<CS: ConvSpec> {
    type TileMatmul: tile::Matmul<CS::ES, CS::EA> + MatmulKernel;

    type StageSize: StageSize;
    type StageMatmul: stage::Matmul<CS::ES, CS::EG, CS::EA> + MatmulKernel;

    type GlobalConvolution: Convolution<CS, Self::StageMatmul> + ConvolutionLaunch<CS::EG, CS::EG>;

    /// Cube dim for launch
    fn cube_dim() -> CubeDim;
    /// The cube count for a given convolution problem
    fn cube_count(problem: &ConvolutionProblem) -> CubeCount;

    /// Make a convolution config from a convolution problem, and launch options
    fn make_config(
        problem: &ConvolutionProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> <Self::GlobalConvolution as ConvolutionKernel<CS::EG, CS::EG>>::Config {
        Self::GlobalConvolution::make_config(problem, cube_dim, cube_count, advanced_config)
    }

    /// Check availability of the matmul algorithm
    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(), MatmulAvailabilityError> {
        Self::GlobalConvolution::check_availability::<R>(client)
    }

    /// Determine whether the given convolution problem is valid to launch (within hardware limits)
    fn can_launch<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        problem: &ConvolutionProblem,
    ) -> bool {
        if problem.options.groups > 1 || Self::check_availability::<R>(client).is_err() {
            return false;
        }

        let cube_count = Self::cube_count(problem);
        let (max_x, max_y, max_z) = R::max_cube_count();
        match cube_count {
            CubeCount::Static(x, y, z) => x <= max_x && y <= max_y && z <= max_z,
            _ => true,
        }
    }
}

/// Cmma convolution
pub struct Cmma<CS: ConvSpec, Stage: StageSize> {
    pub _cp: PhantomData<CS>,
    pub _stage: PhantomData<Stage>,
}

impl<CS: ConvSpec, Stage: StageSize> Algorithm<CS> for Cmma<CS, Stage> {
    type TileMatmul = Accelerated16x16x16<CS::ES, CS::EA>;
    type StageSize = Stage;
    type StageMatmul =
        stage::multi_buffer::Matmul<CS::ES, CS::EG, CS::EA, Self::TileMatmul, Self::StageSize>;

    type GlobalConvolution = ImplicitGemmConvolution<CS, Self::StageMatmul>;

    fn cube_dim() -> CubeDim {
        CubeDim::new(CS::PLANE_DIM, Self::StageSize::NUM_M, 1)
    }

    fn cube_count(problem: &ConvolutionProblem) -> CubeCount {
        let m_stage = Self::StageSize::NUM_M * Self::TileMatmul::M;
        let n_stage = Self::StageSize::NUM_N * Self::TileMatmul::N;
        let cubes_needed_m = (problem.m as u32).div_ceil(m_stage);
        let cubes_needed_n = (problem.n as u32).div_ceil(n_stage);

        CubeCount::Static(cubes_needed_m, cubes_needed_n, 1)
    }
}
