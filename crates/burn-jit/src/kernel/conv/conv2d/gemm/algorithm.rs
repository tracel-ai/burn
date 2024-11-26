use std::marker::PhantomData;

use cubecl::{
    linalg::matmul::{
        components::{
            stage::{self, StageSize},
            tile::{self, accelerated::Accelerated16x16x16, Matmul as _},
            MatmulKernel,
        },
        kernels::matmul::AdvancedConfig,
    },
    prelude::*,
};

use super::{
    base::{Convolution, ConvolutionKernel, ConvolutionLaunch, ConvolutionProblem},
    homogeneous::base::ImplicitGemmConvolution,
};
use half::f16;

/// Specifications for a matmul algorithm
pub trait Algorithm<EG: Numeric> {
    const PLANE_DIM: u32;

    type EG: Numeric;
    type ES: Numeric;
    type EA: Numeric;

    type TileMatmul: tile::Matmul<Self::ES, Self::EA> + MatmulKernel<Self::ES, Self::EA>;

    type StageSize: StageSize;
    type StageMatmul: stage::Matmul<Self::ES, Self::EG, Self::EA> + MatmulKernel<Self::ES, Self::EG>;

    type GlobalMatmul: Convolution<Self::EG, Self::ES, Self::EA, Self::StageMatmul>
        + ConvolutionLaunch<Self::EG, Self::EG>;

    fn cube_dim() -> CubeDim;
    fn cube_count(problem: &ConvolutionProblem) -> CubeCount;

    fn make_config(
        problem: &ConvolutionProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> <Self::GlobalMatmul as ConvolutionKernel<Self::EG, Self::EG>>::Config {
        Self::GlobalMatmul::make_config(problem, cube_dim, cube_count, advanced_config)
    }

    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(), &str> {
        Self::GlobalMatmul::check_availability::<R>(client)
    }

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

pub struct Cmma<EG: Numeric, Stage: StageSize> {
    pub _eg: PhantomData<EG>,
    pub _stage: PhantomData<Stage>,
}

impl<EG: Numeric, Stage: StageSize> Algorithm<EG> for Cmma<EG, Stage> {
    const PLANE_DIM: u32 = 32;
    type EG = EG;
    type ES = half::f16;
    type EA = f32;

    type TileMatmul = Accelerated16x16x16<Self::ES, Self::EA>;

    type StageSize = Stage;
    type StageMatmul = stage::multi_buffer::Matmul<
        Self::ES,
        Self::EG,
        Self::EA,
        Self::TileMatmul,
        Self::StageSize,
    >;

    type GlobalMatmul = ImplicitGemmConvolution<Self::EG, Self::ES, Self::EA, Self::StageMatmul>;

    fn cube_dim() -> CubeDim {
        CubeDim::new(Self::PLANE_DIM, Self::StageSize::NUM_M, 1)
    }

    fn cube_count(problem: &ConvolutionProblem) -> CubeCount {
        let m_stage = Self::StageSize::NUM_M * Self::TileMatmul::M;
        let n_stage = Self::StageSize::NUM_N * Self::TileMatmul::N;
        let cubes_needed_m = (problem.m as u32).div_ceil(m_stage);
        let cubes_needed_n = (problem.n as u32).div_ceil(n_stage);

        CubeCount::Static(cubes_needed_m, cubes_needed_n, 1)
    }
}

pub struct CmmaHalf<EG: Numeric, Stage: StageSize> {
    pub _eg: PhantomData<EG>,
    pub _stage: PhantomData<Stage>,
}

impl<EG: Numeric, Stage: StageSize> Algorithm<EG> for CmmaHalf<EG, Stage> {
    const PLANE_DIM: u32 = 32;
    type EG = EG;
    type ES = f16;
    type EA = f16;

    type TileMatmul = Accelerated16x16x16<Self::ES, Self::EA>;

    type StageSize = Stage;
    type StageMatmul = stage::multi_buffer::Matmul<
        Self::ES,
        Self::EG,
        Self::EA,
        Self::TileMatmul,
        Self::StageSize,
    >;

    type GlobalMatmul = ImplicitGemmConvolution<Self::EG, Self::ES, Self::EA, Self::StageMatmul>;

    fn cube_dim() -> CubeDim {
        CubeDim::new(Self::PLANE_DIM, Self::StageSize::NUM_M, 1)
    }

    fn cube_count(problem: &ConvolutionProblem) -> CubeCount {
        let m_stage = Self::StageSize::NUM_M * Self::TileMatmul::M;
        let n_stage = Self::StageSize::NUM_N * Self::TileMatmul::N;
        let cubes_needed_m = (problem.m as u32).div_ceil(m_stage);
        let cubes_needed_n = (problem.n as u32).div_ceil(n_stage);

        CubeCount::Static(cubes_needed_m, cubes_needed_n, 1)
    }
}
