use config::HomogeneousConfig;
use cubecl::{
    linalg::{
        matmul::{
            components::{
                global::{
                    self,
                    output_loader::Unloader,
                    single_stage::{self, loader::RhsLoader, CyclicLoading},
                    AccumulatorLoader, GlobalConfig, InputLoader,
                },
                stage::{
                    self,
                    multi_buffer::{LhsReader, LhsReaderFamily, RhsReader, RhsReaderFamily},
                    StageMatmulFamily, TilingLayout,
                },
                Ident, InvalidConfigError, MatrixLayout, StageTiling,
            },
            kernels::matmul::AdvancedConfig,
        },
        tensor::{ReadWrite, VirtualTensor},
    },
    prelude::*,
};
use std::marker::PhantomData;

use crate::kernel::conv::{
    conv2d::gemm::base::{
        Convolution, ConvolutionConfigFactory, ConvolutionFamily, ConvolutionLaunch,
        ConvolutionProblem,
    },
    loader::im2col::SimpleIm2colLoader,
    precision::ConvPrecision,
};
use crate::kernel::conv::{conv2d::gemm::ConvGemmConfig as _, loader::bias::BiasLoader};

pub struct ImplicitGemmConvolutionFamily<SMM: StageMatmulFamily> {
    _smm: PhantomData<SMM>,
}

impl<SMM> ConvolutionFamily<SMM> for ImplicitGemmConvolutionFamily<SMM>
where
    SMM: StageMatmulFamily<LhsReader = LhsReaderFamily, RhsReader = RhsReaderFamily>,
{
    type Convolution<CS: ConvPrecision> =
        ImplicitGemmConvolution<CS, SMM::Matmul<CS::ES, CS::EG, CS::EA>>;
}

/// Performs matrix multiplication at the global level, with each plane sharing the same responsibilities
/// - All planes load data to the stage
/// - All planes are used in the stage matmul computation
pub struct ImplicitGemmConvolution<
    CS: ConvPrecision,
    SMM: stage::StageMatmul<CS::ES, CS::EG, CS::EA>,
> {
    _cs: PhantomData<CS>,
    _stage_matmul: PhantomData<SMM>,
}

#[cube]
impl<CS: ConvPrecision, SMM> Convolution<CS, SMM> for ImplicitGemmConvolution<CS, SMM>
where
    SMM: stage::StageMatmul<
        CS::ES,
        CS::EG,
        CS::EA,
        LhsReader = LhsReader<CS::ES>,
        RhsReader = RhsReader<CS::ES>,
    >,
{
    type LhsLoader = SimpleIm2colLoader<CS, Self::Config>;
    type Config = HomogeneousConfig<single_stage::Config<SMM::Config>>;
    type RhsLoader = RhsLoader<CS::EG, CS::ES, SMM::Config, CyclicLoading>;
    type AccumulatorLoader = BiasLoader<CS, SMM::Config>;

    type Out = Unloader<CS::EG>;
    type Accumulator = SMM::Accumulator;

    fn execute(
        mut lhs_loader: Self::LhsLoader,
        mut rhs_loader: Self::RhsLoader,
        mut acc_loader: Self::AccumulatorLoader,
        mut out_unloader: Self::Out,
        acc: &mut Self::Accumulator,
        k_range: (u32, u32),
        #[comptime] config: Self::Config,
    ) {
        let k_step = config.k_step;
        let range = k_range.1 - k_range.0;
        #[allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
        #[allow(clippy::manual_div_ceil)]
        let num_loops = (range + k_step - 1) / k_step;

        Self::AccumulatorLoader::fill_stage(&mut acc_loader, config.to_smm_config());
        let (mut lhs_tile, mut rhs_tile) = SMM::init_tile_inputs(config.to_smm_config());

        sync_units();

        SMM::fill_accumulator::<Self::AccumulatorLoader>(
            &mut acc_loader,
            acc,
            config.to_smm_config(),
        );

        for _ in 0..num_loops {
            sync_units();

            Self::LhsLoader::fill_stage(&mut lhs_loader, config);
            Self::RhsLoader::fill_stage(&mut rhs_loader, config.to_matmul_config());

            let lhs_stage_reader = &Self::LhsLoader::as_stage_reader(&lhs_loader);
            let rhs_stage_reader = &Self::RhsLoader::as_stage_reader(&rhs_loader);

            sync_units();

            SMM::execute(
                lhs_stage_reader,
                rhs_stage_reader,
                &mut lhs_tile,
                &mut rhs_tile,
                acc,
                config.to_smm_config(),
            );

            Self::LhsLoader::advance_view(&mut lhs_loader, k_step);
            Self::RhsLoader::advance_view(&mut rhs_loader, k_step);
        }

        sync_units();

        SMM::read_accumulator::<Self::Out, Self::Config>(
            acc,
            &mut out_unloader,
            config.to_smm_config(),
            config,
        );
    }

    fn init_lhs_loader(
        lhs: VirtualTensor<CS::EG>,
        x_offset: u32,
        y_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::LhsLoader {
        Self::LhsLoader::new(
            lhs,
            config.out_shape(0),
            config.out_shape(1),
            x_offset,
            y_offset,
            config,
        )
    }

    fn init_rhs_loader(
        rhs: VirtualTensor<CS::EG>,
        x_offset: u32,
        y_offset: u32,
        #[comptime] config: Self::Config,
    ) -> Self::RhsLoader {
        Self::RhsLoader::new::<Self::Config>(rhs, x_offset, y_offset, 0, config)
    }

    fn init_bias_loader(
        bias: VirtualTensor<CS::EG>,
        n_offset: u32,
        #[comptime] config: Self::Config,
        #[comptime] has_bias: bool,
    ) -> Self::AccumulatorLoader {
        Self::AccumulatorLoader::new(bias, n_offset, config.to_smm_config(), has_bias)
    }

    fn init_unloader(
        out: VirtualTensor<CS::EG, ReadWrite>,
        x_offset: u32,
        y_offset: u32,
    ) -> Self::Out {
        Self::Out::new(out, x_offset, y_offset, 0)
    }

    fn init_accumulator(#[comptime] config: Self::Config) -> Self::Accumulator {
        SMM::init_accumulator(config.to_smm_config())
    }
}

impl<SMM> ConvolutionConfigFactory for ImplicitGemmConvolutionFamily<SMM>
where
    SMM: StageMatmulFamily,
{
    type Config = config::HomogeneousConfig<single_stage::Config<SMM::Config>>;
    type Input = SMM::Input;

    fn check_config(config: &Self::Config) -> Result<(), InvalidConfigError> {
        SMM::check_config(&config.to_smm_config())
    }

    fn make_config(
        input: Self::Input,
        problem: &ConvolutionProblem,
        cube_dim: &CubeDim,
        cube_count: &CubeCount,
        advanced_config: &AdvancedConfig,
    ) -> Self::Config {
        let smm_config = SMM::make_config(
            input,
            &problem.as_matmul_problem(),
            cube_dim,
            cube_count,
            advanced_config,
            false,
        );
        let size = SMM::stage_shape(&smm_config);

        config::HomogeneousConfig::new(
            single_stage::Config::new(
                smm_config,
                // TODO: Find the correct condition to avoid check bounds.
                true,
                true,
                true,
                problem.lhs_layout,
                problem.rhs_layout,
                problem.lhs_line_size as u32,
                problem.rhs_line_size as u32,
                problem.out_line_size as u32,
                size.k,
                global::LoadMode::Coalesced,
            ),
            (problem.out_shape_y as u32, problem.out_shape_x as u32),
            problem.kernel_size,
            &problem.options,
            problem.has_bias,
        )
    }
}

impl<SMM: StageMatmulFamily<LhsReader = LhsReaderFamily, RhsReader = RhsReaderFamily>>
    ConvolutionLaunch for ImplicitGemmConvolutionFamily<SMM>
{
    unsafe fn launch_unchecked<CS: ConvPrecision, R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount,
        input: TensorArg<'_, R>,
        weight: TensorArg<'_, R>,
        bias: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        config: <Self as ConvolutionConfigFactory>::Config,
    ) {
        implicit_conv::launch_unchecked::<CS::EG, CS::ES, CS::EA, Self, SMM, R>(
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
pub(crate) fn implicit_conv<
    EG: Numeric,
    ES: Numeric,
    EA: Numeric,
    GMM: ConvolutionFamily<SMM>,
    SMM: StageMatmulFamily,
>(
    lhs: &Tensor<Line<EG>>,
    rhs: &Tensor<Line<EG>>,
    bias: &Tensor<Line<EG>>,
    out: &mut Tensor<Line<EG>>,
    #[comptime] config: GMM::Config,
    #[comptime] has_bias: bool,
) {
    let x_offset = CUBE_POS_X * config.stage_tiling(Ident::Lhs).total_row();
    let y_offset = CUBE_POS_Y * config.stage_tiling(Ident::Rhs).total_col();
    let k_range = (0, rhs.shape(0));

    let lhs = VirtualTensor::<EG>::new::<Tensor<Line<EG>>>(lhs);
    let rhs = VirtualTensor::<EG>::new::<Tensor<Line<EG>>>(rhs);
    let bias = VirtualTensor::<EG>::new::<Tensor<Line<EG>>>(bias);
    let out = VirtualTensor::<EG, ReadWrite>::new::<Tensor<Line<EG>>>(out);

    GMM::Convolution::<(EG, ES, EA)>::execute(
        GMM::Convolution::<(EG, ES, EA)>::init_lhs_loader(lhs, x_offset, k_range.0, config),
        GMM::Convolution::<(EG, ES, EA)>::init_rhs_loader(rhs, k_range.0, y_offset, config),
        GMM::Convolution::<(EG, ES, EA)>::init_bias_loader(bias, y_offset, config, has_bias),
        GMM::Convolution::<(EG, ES, EA)>::init_unloader(out, x_offset, y_offset),
        &mut GMM::Convolution::<(EG, ES, EA)>::init_accumulator(config),
        k_range,
        config,
    );
}

pub mod config {
    use std::ops::Deref;

    use burn_tensor::ops::ConvOptions;
    use cubecl::linalg::matmul::components::MatmulConfig;
    use global::GlobalConfig;

    use crate::kernel::conv::conv2d::gemm::{self};

    use super::*;

    #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
    pub struct HomogeneousConfig<M: GlobalConfig> {
        matmul: M,
        out_shape: (u32, u32),
        kernel_size: (u32, u32),
        stride: (u32, u32),
        dilation: (u32, u32),
        padding: (i32, i32),
        pub has_bias: bool,
    }

    impl<M: GlobalConfig> Deref for HomogeneousConfig<M> {
        type Target = M;

        fn deref(&self) -> &Self::Target {
            &self.matmul
        }
    }

    impl<M: GlobalConfig> GlobalConfig for HomogeneousConfig<M> {
        type SmmConfig = M::SmmConfig;

        fn to_smm_config(&self) -> Self::SmmConfig {
            self.matmul.to_smm_config()
        }

        fn global_line_size(&self, ident: Ident) -> u32 {
            self.matmul.global_line_size(ident)
        }

        fn stage_line_size(&self, ident: Ident) -> u32 {
            self.matmul.stage_line_size(ident)
        }

        fn stage_tiling(&self, ident: Ident) -> StageTiling {
            self.matmul.stage_tiling(ident)
        }

        fn layout(&self, ident: Ident) -> MatrixLayout {
            self.matmul.layout(ident)
        }

        fn num_planes(&self) -> u32 {
            self.matmul.num_planes()
        }

        fn plane_dim(&self) -> u32 {
            self.matmul.plane_dim()
        }

        fn tiling_layout(&self, ident: Ident) -> TilingLayout {
            self.matmul.tiling_layout(ident)
        }

        fn check_row_bounds(&self, ident: Ident) -> bool {
            self.matmul.check_row_bounds(ident)
        }

        fn check_col_bounds(&self, ident: Ident) -> bool {
            self.matmul.check_col_bounds(ident)
        }

        fn transpose_load(&self, ident: Ident) -> bool {
            self.matmul.transpose_load(ident)
        }

        fn load_mode(&self) -> global::LoadMode {
            self.matmul.load_mode()
        }
    }

    impl<M: GlobalConfig> gemm::ConvGemmConfig for HomogeneousConfig<M> {
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
    }

    impl<M: GlobalConfig> MatmulConfig for HomogeneousConfig<M> {}

    impl<M: GlobalConfig> HomogeneousConfig<M> {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            matmul: M,
            out_shape: (u32, u32),
            kernel_size: (u32, u32),
            conv_args: &ConvOptions<2>,
            has_bias: bool,
        ) -> Self {
            Self {
                matmul,
                out_shape,
                kernel_size,
                stride: (conv_args.stride[0] as u32, conv_args.stride[1] as u32),
                dilation: (conv_args.dilation[0] as u32, conv_args.dilation[1] as u32),
                padding: (conv_args.padding[0] as i32, conv_args.padding[1] as i32),
                has_bias,
            }
        }

        pub fn to_matmul_config(self) -> M {
            self.matmul
        }
    }
}
