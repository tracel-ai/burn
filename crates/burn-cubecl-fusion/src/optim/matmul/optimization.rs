use super::args::FusedMatmulInputLaunch;
#[cfg(feature = "autotune")]
use super::tune::fused_matmul_autotune;
use crate::{
    CubeFusionHandle, FallbackOperation,
    engine::{
        codegen::ir::{FuseArg, FuseBlockConfig, FuseType, GlobalArgsLaunch, RefLayout},
        launch::{
            FuseTraceLauncher, HandleInput, LaunchPlan,
            runner::{TraceRunner, Vectorization, VectorizationAxis},
        },
        trace::{FuseTrace, TraceError, TuneOutput},
    },
    optim::{
        elemwise::ElemwiseRunner,
        matmul::args::{FusedMatmulArgs, MatmulArg},
    },
};
use burn_fusion::stream::Context;
use burn_ir::BinaryOpIr;
use cubecl::{
    client::ComputeClient,
    prelude::*,
    std::tensor::{MatrixBatchLayout, matrix_batch_layout},
};
use cubek::matmul::{
    components::tile::{cmma::CmmaMatmul, io::Filled, mma::MmaMatmul},
    definition::{
        MatmulElemType, MatmulElems, MatmulLineSizes, MatmulProblem, MatmulSetupError, MatrixLayout,
    },
    launch::launch_kernel_virtual,
    routines::{
        BlueprintStrategy, Routine,
        double_buffering::{CyclicDoubleBufferingAlgorithm, DoubleBufferingArgs},
        double_unit::DoubleUnitAlgorithm,
        ordered_double_buffering::{OrderedDoubleBufferingAlgorithm, OrderedSelectionArgs},
        simple::{SimpleAlgorithm, SimpleArgs},
        simple_unit::SimpleUnitAlgorithm,
        vecmat::{DoubleVecMatAlgorithm, SimpleVecMatAlgorithm},
    },
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Fuse matmul operation followed by elemwise operations into a single kernel.
pub struct MatmulOptimization<R: Runtime> {
    pub(crate) info: Arc<MatmulOptimizationInfo<R>>,
}

pub struct MatmulOptimizationTuneArg<R: Runtime> {
    pub(crate) info: Arc<MatmulOptimizationInfo<R>>,
    pub(crate) fallback: Box<dyn FallbackOperation<R>>,
}

pub(crate) struct MatmulOptimizationInfo<R: Runtime> {
    trace: FuseTrace,
    trace_fallback: FuseTrace,
    pub(crate) client: ComputeClient<R>,
    pub(crate) device: R::Device,
    pub(crate) len: usize,
    pub(crate) matmul: FusedMatmul,
}

#[derive(Serialize, Deserialize, Debug)]
/// State for the [matrix optimization](MatmulOptimizationState).
pub struct MatmulOptimizationState {
    trace: FuseTrace,
    trace_fallback: FuseTrace,
    matmul: FusedMatmul,
    len: usize,
}

impl<R: Runtime> MatmulOptimizationInfo<R> {
    /// Returns the number of output buffers added by fusion.
    pub fn num_output_buffers(&self) -> usize {
        self.trace_fallback.resources.outputs.len()
    }

    /// Number of operations fused.
    pub fn num_ops_fused(&self) -> usize {
        self.len
    }
}

impl<R: Runtime> MatmulOptimizationTuneArg<R> {
    pub(crate) fn execute_fused<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        selector: FusedMatmulSelector,
    ) -> Result<TuneOutput<R>, TraceError<FusedMatmulError>> {
        let launch = FusedMatmulLaunch::new(&self.info.matmul, selector);
        let launcher = FuseTraceLauncher::new(&self.info.trace, &launch);

        launcher.launch::<BT>(&self.info.client, &self.info.device, context)
    }

    pub fn execute_fallback<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> TuneOutput<R> {
        self.fallback.run(context);

        #[cfg(feature = "autotune-checks")]
        let mut output = TuneOutput::Checked {
            handles: Default::default(),
        };
        #[cfg(not(feature = "autotune-checks"))]
        let output = TuneOutput::UnChecked(core::marker::PhantomData);

        #[cfg(feature = "autotune-checks")]
        if let TuneOutput::Checked { handles } = &mut output {
            let out_desc = context.tensors.get(&self.info.matmul.op.out.id).unwrap();
            let handle_out = context
                .handles
                .get_handle(&out_desc.id, &burn_ir::TensorStatus::ReadOnly);

            handles.insert(
                self.info.matmul.op.out.id,
                (out_desc.shape.dims.clone(), handle_out.clone()),
            );
        }

        let launcher = FuseTraceLauncher::new(&self.info.trace_fallback, &ElemwiseRunner);
        let output_write = launcher
            .launch::<BT>(&self.info.client, &self.info.device, context)
            .unwrap();

        output.merge(output_write)
    }
}

impl<R: Runtime> MatmulOptimization<R> {
    pub fn new(
        trace: FuseTrace,
        trace_fallback: FuseTrace,
        client: ComputeClient<R>,
        device: R::Device,
        len: usize,
        matmul: FusedMatmul,
    ) -> Self {
        let info = MatmulOptimizationInfo {
            trace,
            trace_fallback,
            client,
            device,
            len,
            matmul,
        };

        Self {
            info: Arc::new(info),
        }
    }
    /// Execute the optimization.
    pub fn execute<BT: CubeElement>(
        &mut self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        fallback: impl FnOnce(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        // The index of the fallback matmul is always 0.
        let fallback = fallback(0);
        let arg = MatmulOptimizationTuneArg {
            info: self.info.clone(),
            fallback,
        };

        #[cfg(feature = "autotune")]
        fused_matmul_autotune::<R, BT>(arg, context);

        #[cfg(not(feature = "autotune"))]
        if arg
            .execute_fused::<BT>(context, FusedMatmulSelector::default())
            .is_err()
        {
            arg.execute_fallback::<BT>(context);
        }
    }

    /// Number of operations fused.
    pub fn num_ops_fused(&self) -> usize {
        self.info.num_ops_fused()
    }

    /// Create an optimization from its [state](MatmulOptimizationState).
    pub fn from_state(device: &R::Device, state: MatmulOptimizationState) -> Self {
        let info = MatmulOptimizationInfo {
            trace: state.trace,
            trace_fallback: state.trace_fallback,
            len: state.len,
            client: R::client(device),
            device: device.clone(),
            matmul: state.matmul.clone(),
        };

        Self {
            info: Arc::new(info),
        }
    }

    /// Convert the optimization to its [state](MatmulOptimizationState).
    pub fn to_state(&self) -> MatmulOptimizationState {
        MatmulOptimizationState {
            trace: self.info.trace.clone(),
            trace_fallback: self.info.trace_fallback.clone(),
            matmul: self.info.matmul.clone(),
            len: self.info.len,
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub enum FusedMatmulSelector {
    Simple {
        multi_rows: bool,
        tile_matmul: AcceleratedTileKind,
    },
    DoubleBuffering {
        specialized: bool,
        tile_matmul: AcceleratedTileKind,
    },
    OrderedDoubleBuffering {
        tile_matmul: AcceleratedTileKind,
    },
    SimpleVecMat,
    DoubleVecMat,
    SimpleUnit,
    DoubleUnit,
}

impl FusedMatmulSelector {
    /// Not efficient, but only called once when initializing the tunables.
    pub fn name(&self) -> String {
        let name = match self {
            FusedMatmulSelector::Simple {
                multi_rows,
                tile_matmul,
            } => match multi_rows {
                false => format!("simple_{tile_matmul:?}"),
                true => format!("simple_multirows_{tile_matmul:?}"),
            },
            FusedMatmulSelector::DoubleBuffering {
                specialized,
                tile_matmul,
            } => match specialized {
                false => format!("double_buffering_{tile_matmul:?}"),
                true => format!("double_buffering_specialized_{tile_matmul:?}"),
            },
            FusedMatmulSelector::OrderedDoubleBuffering { tile_matmul } => {
                format!("double_buffering_ordered_{tile_matmul:?}").to_lowercase()
            }
            FusedMatmulSelector::SimpleVecMat => "simple_vec_mat".into(),
            FusedMatmulSelector::DoubleVecMat => "double_buffering_vec_mat".into(),
            FusedMatmulSelector::SimpleUnit => "simple_unit".into(),
            FusedMatmulSelector::DoubleUnit => "double_buffering_unit".into(),
        };

        format!("fused_{name}")
    }
}

impl Default for FusedMatmulSelector {
    fn default() -> Self {
        FusedMatmulSelector::Simple {
            multi_rows: false,
            tile_matmul: AcceleratedTileKind::Cmma,
        }
    }
}

#[derive(new, Clone, Serialize, Deserialize, Debug)]
pub struct FusedMatmul {
    lhs: MatmulArg,
    rhs: MatmulArg,
    out: FuseArg,
    pub(crate) op: BinaryOpIr,
    pub(crate) selector: FusedMatmulSelector,
}

#[derive(new)]
pub struct FusedMatmulLaunch<'a> {
    pub(crate) matmul: &'a FusedMatmul,
    pub(crate) selector: FusedMatmulSelector,
}

#[derive(Debug)]
pub enum FusedMatmulError {
    LaunchError(MatmulSetupError),
    InvalidInput(&'static str),
}

impl From<MatmulSetupError> for FusedMatmulError {
    fn from(value: MatmulSetupError) -> Self {
        Self::LaunchError(value)
    }
}

impl<'a, R: Runtime> Vectorization<R> for FusedMatmulLaunch<'a> {
    fn axis(&self, plan: &LaunchPlan<'_, R>) -> VectorizationAxis {
        let lhs_id = self.matmul.op.lhs.id;
        let rhs_id = self.matmul.op.rhs.id;

        let mut tensor_lhs = None;
        let mut tensor_rhs = None;

        for input in plan.handle_inputs.iter() {
            match input {
                HandleInput::Normal(input) => {
                    if input.relative_id == lhs_id {
                        tensor_lhs = Some((input.global_ir.id, &input.handle.strides));
                    }
                    if input.relative_id == rhs_id {
                        tensor_rhs = Some((input.global_ir.id, &input.handle.strides));
                    }
                }
                HandleInput::QuantValues(input) => {
                    if input.relative_id == lhs_id {
                        tensor_lhs = Some((input.global_ir.id, &input.handle.strides));
                    }
                    if input.relative_id == rhs_id {
                        tensor_rhs = Some((input.global_ir.id, &input.handle.strides));
                    }
                }
                HandleInput::QuantParams(_) => {}
            }
        }

        let (lhs_id_global, lhs_strides) = tensor_lhs.unwrap();
        let (rhs_id_global, rhs_strides) = tensor_rhs.unwrap();

        let mut axis = VectorizationAxis::default();

        if let MatrixBatchLayout::MildlyPermuted { transposed, .. } =
            matrix_batch_layout(lhs_strides)
            && transposed
        {
            axis.insert(lhs_id_global, lhs_strides.len() - 2);
        }

        if let MatrixBatchLayout::MildlyPermuted { transposed, .. } =
            matrix_batch_layout(rhs_strides)
            && transposed
        {
            axis.insert(rhs_id_global, rhs_strides.len() - 2);
        }

        axis
    }
}

impl<R: Runtime> TraceRunner<R> for FusedMatmulLaunch<'_> {
    type Error = FusedMatmulError;

    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        configs: &'a [FuseBlockConfig],
    ) -> Result<(), FusedMatmulError> {
        let (lhs, rhs, out) = (
            MatmulElemType {
                dtype: self.matmul.lhs.precision().into_type(),
                quantized: false,
            },
            MatmulElemType {
                dtype: self.matmul.rhs.precision().into_type(),
                quantized: false,
            },
            MatmulElemType {
                dtype: self.matmul.out.precision().into_type(),
                quantized: false,
            },
        );
        let dtypes = MatmulElems::from_globals(lhs, rhs, out);
        self.matmul_fused(client, inputs, outputs, &configs[0], dtypes)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Which tile matmul to use for accelerated algorithms
pub enum AcceleratedTileKind {
    #[default]
    Cmma,
    Mma,
}

macro_rules! with_tile_kind {
    ($kind: expr, $T: ident, $launch: expr) => {
        match $kind {
            AcceleratedTileKind::Cmma => {
                type $T = CmmaMatmul<Filled>;
                ($launch)()
            }
            AcceleratedTileKind::Mma => {
                type $T = MmaMatmul;
                ($launch)()
            }
        }
    };
}

impl FusedMatmulLaunch<'_> {
    fn matmul_fused<'a, R: Runtime>(
        &'a self,
        client: &'a ComputeClient<R>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        config: &'a FuseBlockConfig,
        dtypes: MatmulElems,
    ) -> Result<(), FusedMatmulError> {
        let lhs_shape = inputs.shape(self.matmul.lhs.data());
        let rhs_shape = inputs.shape(self.matmul.rhs.data());
        let out_shape = outputs.shape_ref(&config.ref_layout, config.rank as usize);

        let lhs_strides = inputs.strides(self.matmul.lhs.data());
        let rhs_strides = inputs.strides(self.matmul.rhs.data());

        if matrix_batch_layout(&lhs_strides) == MatrixBatchLayout::HighlyPermuted {
            return Err(FusedMatmulError::InvalidInput(
                "Lhs needs to be contiguous, but can't when fusing.",
            ));
        }
        if matrix_batch_layout(&rhs_strides) == MatrixBatchLayout::HighlyPermuted {
            return Err(FusedMatmulError::InvalidInput(
                "Rhs needs to be contiguous, but can't when fusing.",
            ));
        }

        let mut line_sizes = MatmulLineSizes {
            lhs: inputs.line_size(self.matmul.lhs.data()),
            rhs: inputs.line_size(self.matmul.rhs.data()),
            out: match &config.ref_layout {
                RefLayout::Concrete(arg) => match arg {
                    FuseArg::Input(..) => inputs.line_size(arg),
                    FuseArg::Output(..) => outputs.line_size(arg),
                    _ => panic!("Invalid ref layout"),
                },
                RefLayout::Virtual(_) => 1,
            },
        };

        if line_sizes.out == 1 && (line_sizes.lhs > 1 || line_sizes.rhs > 1) {
            return Err(FusedMatmulError::InvalidInput(
                "Output line size of 1 removes the gain from fusion",
            ));
        }

        if let MatmulArg::Quantized { scheme, .. } = self.matmul.lhs {
            line_sizes.lhs *= scheme.num_quants() as u8;
        }
        if let MatmulArg::Quantized { scheme, .. } = self.matmul.rhs {
            line_sizes.rhs *= scheme.num_quants() as u8;
        }

        let out_strides = MatrixLayout::RowMajor.to_strides(&out_shape);
        let problem = MatmulProblem::from_shapes_and_strides(
            lhs_shape,
            rhs_shape,
            out_shape,
            lhs_strides,
            rhs_strides,
            out_strides,
        );

        match self.selector {
            FusedMatmulSelector::Simple {
                multi_rows,
                tile_matmul,
            } => with_tile_kind!(tile_matmul, Accelerated, || match launch_inner_fix_dtype::<
                R,
                SimpleAlgorithm<Accelerated>,
            >(
                client,
                FusedMatmulInputLaunch::new(
                    inputs,
                    config.clone(),
                    self.matmul.lhs.clone(),
                    self.matmul.rhs.clone(),
                    Option::None,
                    self.matmul.out.clone(),
                ),
                outputs,
                problem,
                line_sizes,
                &BlueprintStrategy::Inferred(SimpleArgs { multi_rows }),
                dtypes,
            ) {
                Ok(_) => Ok(()),
                Err(err) => Err(FusedMatmulError::LaunchError(err)),
            }),
            FusedMatmulSelector::DoubleBuffering {
                specialized,
                tile_matmul,
            } => with_tile_kind!(tile_matmul, Accelerated, || match launch_inner_fix_dtype::<
                R,
                CyclicDoubleBufferingAlgorithm<Accelerated>,
            >(
                client,
                FusedMatmulInputLaunch::new(
                    inputs,
                    config.clone(),
                    self.matmul.lhs.clone(),
                    self.matmul.rhs.clone(),
                    Option::None,
                    self.matmul.out.clone(),
                ),
                outputs,
                problem,
                line_sizes,
                &BlueprintStrategy::Inferred(DoubleBufferingArgs { specialized }),
                dtypes,
            ) {
                Ok(_) => Ok(()),
                Err(err) => Err(FusedMatmulError::LaunchError(err)),
            }),
            FusedMatmulSelector::OrderedDoubleBuffering { tile_matmul } => {
                let row_count = match self.matmul.lhs.precision() {
                    FuseType::F16 | FuseType::BF16 => 8,
                    _ => 4,
                };

                with_tile_kind!(tile_matmul, Accelerated, || match launch_inner_fix_dtype::<
                    R,
                    OrderedDoubleBufferingAlgorithm<Accelerated>,
                >(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.matmul.lhs.clone(),
                        self.matmul.rhs.clone(),
                        Option::None,
                        self.matmul.out.clone(),
                    ),
                    outputs,
                    problem,
                    line_sizes,
                    &BlueprintStrategy::Inferred(OrderedSelectionArgs {
                        row_count: Some(row_count),
                        rows_per_plane: Some(2),
                        partition_k: Some(2),
                    }),
                    dtypes,
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                })
            }
            FusedMatmulSelector::SimpleUnit => {
                match launch_inner_fix_dtype::<R, SimpleUnitAlgorithm>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.matmul.lhs.clone(),
                        self.matmul.rhs.clone(),
                        Option::None,
                        self.matmul.out.clone(),
                    ),
                    outputs,
                    problem,
                    line_sizes,
                    &Default::default(),
                    dtypes,
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }
            FusedMatmulSelector::DoubleUnit => {
                match launch_inner_fix_dtype::<R, DoubleUnitAlgorithm>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.matmul.lhs.clone(),
                        self.matmul.rhs.clone(),
                        Option::None,
                        self.matmul.out.clone(),
                    ),
                    outputs,
                    problem,
                    line_sizes,
                    &Default::default(),
                    dtypes,
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }
            FusedMatmulSelector::SimpleVecMat => {
                match launch_inner_fix_dtype::<R, SimpleVecMatAlgorithm>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.matmul.lhs.clone(),
                        self.matmul.rhs.clone(),
                        Option::None,
                        self.matmul.out.clone(),
                    ),
                    outputs,
                    problem,
                    line_sizes,
                    &Default::default(),
                    dtypes,
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }
            FusedMatmulSelector::DoubleVecMat => {
                match launch_inner_fix_dtype::<R, DoubleVecMatAlgorithm>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.matmul.lhs.clone(),
                        self.matmul.rhs.clone(),
                        Option::None,
                        self.matmul.out.clone(),
                    ),
                    outputs,
                    problem,
                    line_sizes,
                    &Default::default(),
                    dtypes,
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }
        }
    }
}

fn launch_inner_fix_dtype<'a, R: Runtime, A: Routine>(
    client: &ComputeClient<R>,
    input: FusedMatmulInputLaunch<'a, R>,
    output: GlobalArgsLaunch<'a, R>,
    problem: MatmulProblem,
    line_sizes: MatmulLineSizes,
    blueprint_strategy: &BlueprintStrategy<A>,
    mut dtypes: MatmulElems,
) -> Result<(), MatmulSetupError> {
    let fix_plane_dim = |plane_dim: u32| {
        // Sometimes the GPU doesn't support plane instructions and doesn't report the
        // plane size, but we can still execute algorithms that don't use plane instructions.
        //
        // In this case, we set a plane size for the selector to work, defaulting to 32 as it
        // is a common plane size.
        if plane_dim == 0 { 32 } else { plane_dim }
    };

    let plane_size = fix_plane_dim(A::select_plane_dim(client));

    launch_kernel_virtual::<FusedMatmulArgs, R, A>(
        client,
        input,
        output,
        problem,
        line_sizes,
        plane_size,
        blueprint_strategy,
        &mut dtypes,
    )
}
