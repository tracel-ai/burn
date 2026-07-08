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
    zspace::{Shape, Strides},
};
use cubek::{
    matmul::{
        components::tile::TileMatmulKind,
        definition::{
            MatmulElems, MatmulGlobalElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes,
        },
        routines::{
            BatchMatmulRoutine, BlueprintStrategy,
            batch::{
                double_buffering::{CyclicDoubleBufferingAlgorithm, DoubleBufferingArgs},
                double_unit::DoubleUnitAlgorithm,
                gemv_innerproduct::{
                    DoubleVecMatInnerProductAlgorithm, VecMatInnerProductAlgorithm,
                },
                ordered_double_buffering::{OrderedDoubleBufferingAlgorithm, OrderedSelectionArgs},
                simple::{SimpleAlgorithm, SimpleArgs},
                simple_unit::SimpleUnitAlgorithm,
            },
            gemm::GemmRoutine,
            gemv_unit_perpendicular::GemvUnitPerpendicularRoutine,
        },
        strategy::launch_kernel_virtual,
    },
    std::MatrixLayout,
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
    pub(crate) fn execute_fused(
        &self,
        context: &mut Context<CubeFusionHandle<R>>,
        selector: FusedMatmulSelector,
    ) -> Result<TuneOutput<R>, TraceError<FusedMatmulError>> {
        let launch = FusedMatmulLaunch::new(&self.info.matmul, selector);
        let launcher = FuseTraceLauncher::new(&self.info.trace, &launch);

        launcher.launch(&self.info.client, &self.info.device, context)
    }

    pub fn execute_fallback(&self, context: &mut Context<CubeFusionHandle<R>>) -> TuneOutput<R> {
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
                (out_desc.shape.clone(), handle_out.clone()),
            );
        }

        let launcher = FuseTraceLauncher::new(&self.info.trace_fallback, &ElemwiseRunner);
        let output_write = launcher
            .launch(&self.info.client, &self.info.device, context)
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
    pub fn execute(
        &mut self,
        context: &mut Context<CubeFusionHandle<R>>,
        fallback: impl FnOnce(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        // The index of the fallback matmul is always 0.
        let fallback = fallback(0);
        let arg = MatmulOptimizationTuneArg {
            info: self.info.clone(),
            fallback,
        };

        #[cfg(feature = "autotune")]
        fused_matmul_autotune::<R>(arg, context);

        #[cfg(not(feature = "autotune"))]
        if arg
            .execute_fused(context, FusedMatmulSelector::default())
            .is_err()
        {
            arg.execute_fallback(context);
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
    GemmNoStage,
    GemvUnitPerpendicular,
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
            FusedMatmulSelector::GemmNoStage => "gemm".into(),
            FusedMatmulSelector::GemvUnitPerpendicular => "gemv_unit_perpendicular".into(),
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
    pub(crate) lhs: MatmulArg,
    pub(crate) rhs: MatmulArg,
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
            matrix_batch_layout(lhs_strides, self.matmul.lhs.scheme())
            && transposed
        {
            axis.insert(lhs_id_global, lhs_strides.len() - 2);
        }

        if let MatrixBatchLayout::MildlyPermuted { transposed, .. } =
            matrix_batch_layout(rhs_strides, self.matmul.rhs.scheme())
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
        inputs: GlobalArgsLaunch<R>,
        outputs: GlobalArgsLaunch<R>,
        configs: &'a [FuseBlockConfig],
    ) -> Result<(), FusedMatmulError> {
        let global_elems = MatmulGlobalElems {
            lhs: self.matmul.lhs.precision().into_storage_type(),
            rhs: self.matmul.rhs.precision().into_storage_type(),
            out: self.matmul.out.precision().into_storage_type(),
        };
        let dtypes = MatmulElems::from_globals(&global_elems);
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

impl FusedMatmulLaunch<'_> {
    fn matmul_fused<'a, R: Runtime>(
        &'a self,
        client: &'a ComputeClient<R>,
        inputs: GlobalArgsLaunch<R>,
        outputs: GlobalArgsLaunch<R>,
        config: &'a FuseBlockConfig,
        dtypes: MatmulElems,
    ) -> Result<(), FusedMatmulError> {
        let lhs_shape = inputs.shape(self.matmul.lhs.data());
        let rhs_shape = inputs.shape(self.matmul.rhs.data());
        let out_shape = outputs.shape_ref(&config.ref_layout, config.rank);

        let lhs_strides = inputs.strides(self.matmul.lhs.data());
        let lhs_scheme = self.matmul.lhs.scheme();
        let rhs_strides = inputs.strides(self.matmul.rhs.data());
        let rhs_scheme = self.matmul.rhs.scheme();

        if matrix_batch_layout(&lhs_strides, lhs_scheme) == MatrixBatchLayout::HighlyPermuted {
            return Err(FusedMatmulError::InvalidInput(
                "Lhs needs to be contiguous, but can't when fusing.",
            ));
        }
        if matrix_batch_layout(&rhs_strides, rhs_scheme) == MatrixBatchLayout::HighlyPermuted {
            return Err(FusedMatmulError::InvalidInput(
                "Rhs needs to be contiguous, but can't when fusing.",
            ));
        }

        let mut vector_sizes = MatmulVectorSizes {
            lhs: inputs.vector_size(self.matmul.lhs.data()),
            rhs: inputs.vector_size(self.matmul.rhs.data()),
            out: match &config.ref_layout {
                RefLayout::Concrete(arg) => match arg {
                    FuseArg::Input(..) => inputs.vector_size(arg),
                    FuseArg::Output(..) => outputs.vector_size(arg),
                    _ => panic!("Invalid ref layout"),
                },
                RefLayout::Virtual(_) => 1,
            },
        };

        let address_type = inputs
            .required_address_type()
            .max(outputs.required_address_type());

        if vector_sizes.out == 1 && (vector_sizes.lhs > 1 || vector_sizes.rhs > 1) {
            return Err(FusedMatmulError::InvalidInput(
                "Output vector size of 1 removes the gain from fusion",
            ));
        }

        if let MatmulArg::Quantized { scheme, .. } = self.matmul.lhs {
            vector_sizes.lhs *= scheme.num_quants();
        }
        if let MatmulArg::Quantized { scheme, .. } = self.matmul.rhs {
            vector_sizes.rhs *= scheme.num_quants();
        }

        // When the rhs is broadcast over every batch dim and the lhs/out rows are
        // contiguous across the innermost batch dim, merge that dim into the rows:
        // `[.., b, m, k] @ [.., 1, k, n]` runs as one `[b*m, k] @ [k, n]` matmul
        // instead of `b` broadcast matmuls that each re-read the whole rhs. The
        // views read the merged row dim through the `merged_rows` flag; the fused
        // epilogue is unaffected since every element keeps its linear position.
        let out_strides_ref = outputs.strides_ref(&config.ref_layout, config.rank);
        let merged_rows = matches!(self.matmul.lhs, MatmulArg::Normal(_))
            && can_merge_rows(
                &lhs_shape,
                &lhs_strides,
                &rhs_shape,
                &out_shape,
                &out_strides_ref,
            );

        let (lhs_shape, lhs_strides, out_shape) = match merged_rows {
            true => {
                let (lhs_shape, lhs_strides) = merge_rows(&lhs_shape, &lhs_strides);
                let (out_shape, _) = merge_rows(&out_shape, &out_strides_ref);
                (lhs_shape, lhs_strides, out_shape)
            }
            false => (lhs_shape, lhs_strides, out_shape),
        };

        let out_strides = MatrixLayout::RowMajor.to_strides(&out_shape);
        let problem = MatmulProblem::from_shapes_and_strides(
            lhs_shape,
            rhs_shape,
            out_shape,
            lhs_strides,
            rhs_strides,
            out_strides,
            dtypes.as_global_elems(),
            address_type,
            self.matmul.lhs.scheme(),
            self.matmul.rhs.scheme(),
        )?;

        match self.selector {
            FusedMatmulSelector::Simple {
                multi_rows,
                tile_matmul,
            } => {
                let args = SimpleArgs {
                    multi_rows,
                    tile_matmul: match tile_matmul {
                        AcceleratedTileKind::Cmma => TileMatmulKind::Cmma,
                        AcceleratedTileKind::Mma => TileMatmulKind::Mma,
                    },
                };

                match launch_inner_fix_dtype::<R, SimpleAlgorithm>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.matmul.lhs.clone(),
                        self.matmul.rhs.clone(),
                        None,
                        self.matmul.out.clone(),
                        merged_rows,
                    ),
                    outputs,
                    problem,
                    vector_sizes,
                    &BlueprintStrategy::Inferred(args),
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }

            FusedMatmulSelector::DoubleBuffering {
                specialized,
                tile_matmul,
            } => {
                let args = DoubleBufferingArgs {
                    specialized,
                    tile_matmul: match tile_matmul {
                        AcceleratedTileKind::Cmma => TileMatmulKind::Cmma,
                        AcceleratedTileKind::Mma => TileMatmulKind::Mma,
                    },
                };

                match launch_inner_fix_dtype::<R, CyclicDoubleBufferingAlgorithm>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.matmul.lhs.clone(),
                        self.matmul.rhs.clone(),
                        None,
                        self.matmul.out.clone(),
                        merged_rows,
                    ),
                    outputs,
                    problem,
                    vector_sizes,
                    &BlueprintStrategy::Inferred(args),
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }

            FusedMatmulSelector::OrderedDoubleBuffering { tile_matmul } => {
                let row_count = match self.matmul.lhs.precision() {
                    FuseType::F16 | FuseType::BF16 => 8,
                    _ => 4,
                };

                let args = OrderedSelectionArgs {
                    row_count: Some(row_count),
                    rows_per_plane: Some(2),
                    partition_k: Some(2),
                    tile_matmul: match tile_matmul {
                        AcceleratedTileKind::Cmma => TileMatmulKind::Cmma,
                        AcceleratedTileKind::Mma => TileMatmulKind::Mma,
                    },
                };

                match launch_inner_fix_dtype::<R, OrderedDoubleBufferingAlgorithm>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.matmul.lhs.clone(),
                        self.matmul.rhs.clone(),
                        None,
                        self.matmul.out.clone(),
                        merged_rows,
                    ),
                    outputs,
                    problem,
                    vector_sizes,
                    &BlueprintStrategy::Inferred(args),
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }

            FusedMatmulSelector::SimpleUnit => {
                match launch_inner_fix_dtype::<R, SimpleUnitAlgorithm>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.matmul.lhs.clone(),
                        self.matmul.rhs.clone(),
                        None,
                        self.matmul.out.clone(),
                        merged_rows,
                    ),
                    outputs,
                    problem,
                    vector_sizes,
                    &Default::default(),
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
                        None,
                        self.matmul.out.clone(),
                        merged_rows,
                    ),
                    outputs,
                    problem,
                    vector_sizes,
                    &Default::default(),
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }

            FusedMatmulSelector::SimpleVecMat => {
                match launch_inner_fix_dtype::<R, VecMatInnerProductAlgorithm>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.matmul.lhs.clone(),
                        self.matmul.rhs.clone(),
                        None,
                        self.matmul.out.clone(),
                        merged_rows,
                    ),
                    outputs,
                    problem,
                    vector_sizes,
                    &Default::default(),
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }

            FusedMatmulSelector::DoubleVecMat => {
                match launch_inner_fix_dtype::<R, DoubleVecMatInnerProductAlgorithm>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.matmul.lhs.clone(),
                        self.matmul.rhs.clone(),
                        None,
                        self.matmul.out.clone(),
                        merged_rows,
                    ),
                    outputs,
                    problem,
                    vector_sizes,
                    &Default::default(),
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }

            FusedMatmulSelector::GemmNoStage => {
                match launch_inner_fix_dtype::<R, GemmRoutine>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.matmul.lhs.clone(),
                        self.matmul.rhs.clone(),
                        None,
                        self.matmul.out.clone(),
                        merged_rows,
                    ),
                    outputs,
                    problem,
                    vector_sizes,
                    &Default::default(),
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }

            FusedMatmulSelector::GemvUnitPerpendicular => {
                match launch_inner_fix_dtype::<R, GemvUnitPerpendicularRoutine>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.matmul.lhs.clone(),
                        self.matmul.rhs.clone(),
                        None,
                        self.matmul.out.clone(),
                        merged_rows,
                    ),
                    outputs,
                    problem,
                    vector_sizes,
                    &Default::default(),
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }
        }
    }
}

/// Whether the innermost batch dim of a broadcast-rhs matmul can merge into the
/// row dim (see the `merged_rows` flag on `FusedMatmulInput`).
fn can_merge_rows(
    lhs_shape: &Shape,
    lhs_strides: &Strides,
    rhs_shape: &Shape,
    out_shape: &Shape,
    out_strides: &Strides,
) -> bool {
    let rank = out_shape.num_dims();
    if rank < 3 {
        return false;
    }

    // Every batch dim of the rhs must be broadcast.
    if rhs_shape.to_vec()[..rank - 2].iter().any(|&d| d != 1) {
        return false;
    }
    // Only the innermost batch dim merges; outer ones must be 1.
    if lhs_shape.to_vec()[..rank - 3].iter().any(|&d| d != 1) {
        return false;
    }
    if out_shape.to_vec()[..rank - 3].iter().any(|&d| d != 1) {
        return false;
    }

    let batch = out_shape[rank - 3];
    let rows = out_shape[rank - 2];
    if batch == 1 {
        // Nothing to merge.
        return false;
    }
    // Only the degenerate batched vec-mat merges: with real per-batch rows the
    // batched matmul already tiles well, and measurements show it beats the
    // merged single matmul.
    if rows != 1 {
        return false;
    }
    // The lhs can't be broadcast over the merged rows.
    if lhs_shape[rank - 3] != batch || lhs_shape[rank - 2] != rows {
        return false;
    }

    rows_mergeable(lhs_shape, lhs_strides, rank) && rows_mergeable(out_shape, out_strides, rank)
}

/// Rows advance with a single stride across dims `rank-3` and `rank-2`.
fn rows_mergeable(shape: &Shape, strides: &Strides, rank: usize) -> bool {
    shape[rank - 2] == 1 || strides[rank - 3] == shape[rank - 2] * strides[rank - 2]
}

/// Reinterpret `[.., b, m, k]` as `[.., 1, b*m, k]`.
fn merge_rows(shape: &Shape, strides: &Strides) -> (Shape, Strides) {
    let rank = shape.num_dims();
    let mut shape = shape.clone();
    let mut strides = strides.clone();

    let rows = shape[rank - 3] * shape[rank - 2];
    let stride_row = match shape[rank - 2] == 1 {
        true => strides[rank - 3],
        false => strides[rank - 2],
    };

    shape[rank - 2] = rows;
    shape[rank - 3] = 1;
    strides[rank - 2] = stride_row;
    strides[rank - 3] = rows * stride_row;

    (shape, strides)
}

fn launch_inner_fix_dtype<R: Runtime, A: BatchMatmulRoutine<()>>(
    client: &ComputeClient<R>,
    input: FusedMatmulInputLaunch<R>,
    output: GlobalArgsLaunch<R>,
    problem: MatmulProblem,
    vector_sizes: MatmulVectorSizes,
    blueprint_strategy: &BlueprintStrategy<(), A>,
) -> Result<(), MatmulSetupError> {
    launch_kernel_virtual::<FusedMatmulArgs, R, A>(
        client,
        input,
        output,
        (),
        problem,
        vector_sizes,
        blueprint_strategy,
    )
}
