use cubecl::{
    linalg::matmul::{
        components::{
            global::{
                self, homogeneous::RhsLoader, unloader::Unloader, AccumulatorLoader, Config as _,
            },
            stage::{
                self,
                multi_buffer::{LhsReader, RhsReader},
                TilingOrderConfig,
            },
            Ident, MatrixLayout, StageDim,
        },
        kernels::matmul::AdvancedConfig,
    },
    prelude::*,
};
use std::marker::PhantomData;

use crate::kernel::conv::conv2d::gemm::Config as _;
use crate::kernel::conv::{
    conv2d::gemm::base::{Convolution, ConvolutionKernel, ConvolutionLaunch, ConvolutionProblem},
    loader::weight::WeightLoader,
};

use super::loader::{BiasLoader, Loader, SimpleIm2colLoader};

/// Performs matrix multiplication at the global level, with each plane sharing the same responsibilities
/// - All planes load data to the stage
/// - All planes are used in the stage matmul computation
pub struct ImplicitGemmConvolution<
    EG: Numeric,
    ES: Numeric,
    Acc: Numeric,
    SMM: stage::Matmul<ES, EG>,
> {
    _eg: PhantomData<EG>,
    _es: PhantomData<ES>,
    _acc: PhantomData<Acc>,
    _stage_matmul: PhantomData<SMM>,
}

#[cube]
impl<EG, ES, Acc, SMM> Convolution<EG, ES, Acc> for ImplicitGemmConvolution<EG, ES, Acc, SMM>
where
    EG: Numeric,
    ES: Numeric,
    Acc: Numeric,
    SMM: stage::Matmul<
        ES,
        EG,
        Lhs = LhsReader<ES>,
        Rhs = RhsReader<ES>,
        AccumulatorInit = BiasLoader<EG, Acc>,
    >,
{
    type Lhs = SimpleIm2colLoader<EG, ES>;
    type Rhs = RhsLoader<EG, ES>;
    type Bias = BiasLoader<EG, Acc>;
    type Out = Unloader<EG>;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::Lhs,
        mut rhs_loader: Self::Rhs,
        mut bias_loader: Self::Bias,
        mut out_unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = SMM::K;
        let range = k_range.1 - k_range.0;
        let num_loops = (range + k_step - 1) / k_step;

        Self::Bias::fill_stage::<Self::Config>(&mut bias_loader, config);

        sync_units();

        SMM::fill_accumulator(&mut bias_loader, acc, config.to_smm_config());

        for _ in 0..num_loops {
            let lhs_stage_reader = &Self::Lhs::fill_stage::<Self::Config>(&mut lhs_loader, config);
            let rhs_stage_reader = &Self::Rhs::fill_stage::<Self::Config>(&mut rhs_loader, config);

            sync_units();

            SMM::execute(
                lhs_stage_reader,
                rhs_stage_reader,
                acc,
                config.to_smm_config(),
            );

            sync_units();

            Self::Lhs::advance_view(&mut lhs_loader, k_step);
            Self::Rhs::advance_view(&mut rhs_loader, k_step);
        }

        SMM::read_accumulator::<Self::Out, Self::Config>(
            acc,
            &mut out_unloader,
            config.to_smm_config(),
            config,
        );
    }

    fn init_lhs_loader(
        lhs: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::Lhs {
        Self::Lhs::new::<Self::Config>(
            lhs,
            config.out_shape(0),
            config.out_shape(1),
            x_offset,
            y_offset,
            config,
        )
    }

    fn init_rhs_loader(
        rhs: &Tensor<Line<EG>>,
        x_offset: u32,
        y_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::Rhs {
        Self::Rhs::new::<Self::Config>(rhs, x_offset, y_offset, 0, config)
    }

    fn init_bias_loader(
        bias: &Tensor<Line<EG>>,
        n_offset: u32,
        #[comptime] config: Self::Config,
        #[comptime] has_bias: bool,
    ) -> Self::Bias {
        Self::Bias::new::<Self::Config>(bias, n_offset, config, has_bias)
    }

    fn init_unloader(out: &mut Tensor<Line<EG>>, x_offset: u32, y_offset: u32) -> Self::Out {
        Self::Out::new(out, x_offset, y_offset, 0)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.to_smm_config())
    }

    fn zero_accumulator(acc: &mut Self::Accumulator, #[comptime] config: Self::Config) {
        SMM::zero_accumulator(acc, config.to_smm_config());
    }
}

impl<EG, ES, Acc, SMM> ConvolutionKernel<EG, EG> for ImplicitGemmConvolution<EG, ES, Acc, SMM>
where
    EG: Numeric,
    ES: Numeric,
    Acc: Numeric,
    SMM: stage::Matmul<ES, EG>,
{
    type Config = config::Config<SMM::Config>;

    fn check_config(config: Self::Config) {
        SMM::check_config(config.to_smm_config());
    }

    fn check_availability<R: Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Result<(), &str> {
        SMM::check_availability::<R>(client)
    }

    fn make_config(
        problem: &ConvolutionProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Self::Config {
        let smm_config = SMM::make_config(
            &problem.as_matmul_problem(),
            cube_dim,
            cube_count,
            advanced_config,
        );

        config::Config::new(
            smm_config,
            problem.m as u32 % SMM::M != 0,
            problem.n as u32 % SMM::N != 0,
            problem.k as u32 % SMM::K != 0,
            problem.lhs_layout,
            problem.rhs_layout,
            problem.lhs_line_size as u32,
            problem.rhs_line_size as u32,
            problem.out_line_size as u32,
            (problem.out_shape_y as u32, problem.out_shape_x as u32),
            problem.kernel_size,
            &problem.options,
            problem.has_bias,
        )
    }
}

impl<
        EG: Numeric,
        ES: Numeric,
        Acc: Numeric,
        GMM: stage::Matmul<
            ES,
            EG,
            Lhs = LhsReader<ES>,
            Rhs = RhsReader<ES>,
            AccumulatorInit = BiasLoader<EG, Acc>,
        >,
    > ConvolutionLaunch<EG, EG> for ImplicitGemmConvolution<EG, ES, Acc, GMM>
{
    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: TensorArg<'_, R>,
        weight: TensorArg<'_, R>,
        bias: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        config: <Self as ConvolutionKernel<EG, EG>>::Config,
    ) {
        Self::check_config(config);

        launch::launch_unchecked::<EG, ES, Acc, Self, R>(
            client,
            cube_count,
            cube_dim,
            input,
            weight,
            bias,
            out,
            config,
            config.has_bias,
        );
    }
}

#[cube(launch_unchecked)]
pub(crate) fn launch<EG: Numeric, ES: Numeric, Acc: Numeric, GMM: Convolution<EG, ES, Acc>>(
    lhs: &Tensor<Line<EG>>,
    rhs: &Tensor<Line<EG>>,
    bias: &Tensor<Line<EG>>,
    out: &mut Tensor<Line<EG>>,
    #[comptime] config: GMM::Config,
    #[comptime] has_bias: bool,
) {
    let x_offset = CUBE_POS_X * config.stage_dim(Ident::Lhs).num_elements_x_dim();
    let y_offset = CUBE_POS_Y * config.stage_dim(Ident::Rhs).num_elements_y_dim();
    let k_range = (0, rhs.shape(0));

    GMM::execute(
        GMM::init_lhs_loader(lhs, x_offset, k_range.0, config),
        GMM::init_rhs_loader(rhs, k_range.0, y_offset, config),
        GMM::init_bias_loader(bias, y_offset, config, has_bias),
        GMM::init_unloader(out, x_offset, y_offset),
        &mut GMM::init_accumulator(config),
        k_range,
        config,
    );
}

mod config {
    use burn_tensor::ops::ConvOptions;
    use cubecl::linalg::matmul::components::MatmulConfig;

    use crate::kernel::conv::conv2d::gemm::{self};

    use super::*;

    #[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
    pub struct Config<S: stage::Config> {
        smm_config: S,
        check_m_bounds: bool,
        check_n_bounds: bool,
        check_k_bounds: bool,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        lhs_line_size: u32,
        rhs_line_size: u32,
        out_line_size: u32,

        out_shape: (u32, u32),

        kernel_size: (u32, u32),
        stride: (u32, u32),
        dilation: (u32, u32),
        padding: (i32, i32),

        pub has_bias: bool,
    }

    impl<S: stage::Config> global::Config for Config<S> {
        type SmmConfig = S;

        fn to_smm_config(&self) -> Self::SmmConfig {
            self.smm_config
        }

        fn global_line_size(&self, ident: Ident) -> u32 {
            match ident {
                Ident::Lhs => self.lhs_line_size,
                Ident::Rhs => self.rhs_line_size,
                Ident::Out => self.out_line_size,
            }
        }

        fn stage_line_size(&self, ident: Ident) -> u32 {
            self.smm_config.line_size(ident)
        }

        fn stage_dim(&self, ident: Ident) -> StageDim {
            self.smm_config.stage_dim(ident)
        }

        fn layout(&self, ident: Ident) -> MatrixLayout {
            match ident {
                Ident::Lhs => self.lhs_layout,
                Ident::Rhs => self.rhs_layout,
                Ident::Out => self.smm_config.layout(Ident::Out),
            }
        }

        fn num_planes(&self) -> u32 {
            self.smm_config.num_planes()
        }

        fn plane_dim(&self) -> u32 {
            self.smm_config.plane_dim()
        }

        fn tiling_order(&self, ident: Ident) -> TilingOrderConfig {
            self.smm_config.tiling_order(ident)
        }

        fn check_m_bounds(&self) -> bool {
            self.check_m_bounds
        }

        fn check_n_bounds(&self) -> bool {
            self.check_n_bounds
        }

        fn transpose_load(&self, ident: Ident) -> bool {
            self.layout(ident) != self.smm_config.layout(ident)
        }

        fn num_producers(&self) -> u32 {
            self.num_planes()
        }

        fn num_consumers(&self) -> u32 {
            self.num_planes()
        }
    }

    impl<S: stage::Config> gemm::Config for Config<S> {
        fn out_shape(&self, dim: u32) -> u32 {
            match dim {
                0 => self.out_shape.0,
                1 => self.out_shape.1,
                _ => unreachable!(),
            }
        }

        fn kernel_size(&self, dim: u32) -> u32 {
            match dim {
                0 => self.kernel_size.0,
                1 => self.kernel_size.1,
                _ => unreachable!(),
            }
        }

        fn dilation(&self, dim: u32) -> u32 {
            match dim {
                0 => self.dilation.0,
                1 => self.dilation.1,
                _ => unreachable!(),
            }
        }

        fn stride(&self, dim: u32) -> u32 {
            match dim {
                0 => self.stride.0,
                1 => self.stride.1,
                _ => unreachable!(),
            }
        }

        fn padding(&self, dim: u32) -> i32 {
            match dim {
                0 => self.padding.0,
                1 => self.padding.1,
                _ => unreachable!(),
            }
        }

        fn im2col_unchecked(&self) -> bool {
            !self.check_m_bounds && !self.check_k_bounds && self.padding == (0, 0)
        }

        fn check_k_bounds(&self) -> bool {
            self.check_k_bounds
        }
    }

    impl<S: stage::Config> MatmulConfig for Config<S> {}

    impl<S: stage::Config> Config<S> {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            smm_config: S,
            check_m_bounds: bool,
            check_n_bounds: bool,
            check_k_bounds: bool,
            lhs_layout: MatrixLayout,
            rhs_layout: MatrixLayout,
            lhs_line_size: u32,
            rhs_line_size: u32,
            out_line_size: u32,
            out_shape: (u32, u32),
            kernel_size: (u32, u32),
            conv_args: &ConvOptions<2>,
            has_bias: bool,
        ) -> Self {
            Self {
                smm_config,
                check_m_bounds,
                check_n_bounds,
                check_k_bounds,
                lhs_layout,
                rhs_layout,
                lhs_line_size,
                rhs_line_size,
                out_line_size,
                out_shape,
                kernel_size,
                stride: (conv_args.stride[0] as u32, conv_args.stride[1] as u32),
                dilation: (conv_args.dilation[0] as u32, conv_args.dilation[1] as u32),
                padding: (conv_args.padding[0] as i32, conv_args.padding[1] as i32),
                has_bias,
            }
        }
    }
}
