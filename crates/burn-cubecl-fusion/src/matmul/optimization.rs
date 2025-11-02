use std::any::TypeId;
use std::sync::Arc;

use crate::FallbackOperation;
use crate::elemwise::optimization::ElemwiseRunner;
use crate::shared::ir::FusePrecision;
use crate::shared::ir::RefLayout;
use crate::shared::trace::HandleInput;
use crate::shared::trace::LaunchPlan;
use crate::shared::trace::TraceError;
use crate::shared::trace::TuneOutput;
use crate::shared::trace::Vectorization;
use crate::shared::trace::VectorizationAxis;
use crate::{CubeFusionHandle, matmul::args::MatmulArg};

use burn_fusion::stream::Context;
use burn_ir::BinaryOpIr;
use cubecl::features::TypeUsage;
use cubecl::matmul::components::AccG;
use cubecl::matmul::components::AccS;
use cubecl::matmul::components::tile::io::Filled;
use cubecl::matmul::components::{
    self, LhsG, MatmulProblem, MatmulSetupError, RhsG, RhsS,
    tile::{TileMatmulFamily, accelerated::AcceleratedMatmul},
};
use cubecl::matmul::kernels::layered::Selection;
use cubecl::matmul::kernels::layered::double_buffering::CyclicDoubleBufferingAlgorithm;
use cubecl::matmul::kernels::layered::double_buffering::DoubleBufferingArgs;
use cubecl::matmul::kernels::layered::double_unit::DoubleUnitAlgorithm;
use cubecl::matmul::kernels::layered::launch_kernel_virtual;
use cubecl::matmul::kernels::layered::ordered_double_buffering::OrderedDoubleBufferingAlgorithm;
use cubecl::matmul::kernels::layered::ordered_double_buffering::OrderedSelectionArgs;
use cubecl::matmul::kernels::layered::simple::SimpleAlgorithm;
use cubecl::matmul::kernels::layered::simple::SimpleArgs;
use cubecl::matmul::kernels::layered::simple_unit::SimpleUnitAlgorithm;
use cubecl::matmul::kernels::layered::vecmat::DoubleVecMatAlgorithm;
use cubecl::matmul::kernels::layered::vecmat::SimpleVecMatAlgorithm;
use cubecl::matmul::{
    components::{LhsS, MatmulLineSizes, MatmulPrecision},
    kernels::layered::Algorithm,
};
use cubecl::std::tensor::{MatrixBatchLayout, matrix_batch_layout};
use cubecl::{client::ComputeClient, prelude::*};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

use crate::shared::{
    ir::{Arg, FuseBlockConfig, GlobalArgsLaunch},
    trace::{FuseTrace, TraceRunner},
};

use super::args::FusedMatmulInputLaunch;
use super::spec::FusedMatmulSpec;
use super::tune::fused_matmul_autotune;

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
    pub(crate) client: ComputeClient<R::Server>,
    pub(crate) device: R::Device,
    pub(crate) len: usize,
    pub(crate) variants: MatmulVariants,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct MatmulVariants {
    pub(crate) simple_unit: FusedMatmul,
    pub(crate) simple_vec_mat: FusedMatmul,
    pub(crate) double_vec_mat: FusedMatmul,
    pub(crate) double_unit: FusedMatmul,
    pub(crate) simple: FusedMatmul,
    pub(crate) simple_multi_rows: FusedMatmul,
    pub(crate) double_buffering: FusedMatmul,
    pub(crate) specialized: FusedMatmul,
    pub(crate) ordered: FusedMatmul,
}

#[derive(Serialize, Deserialize, Debug)]
/// State for the [matrix optimization](MatmulOptimizationState).
pub struct MatmulOptimizationState {
    trace: FuseTrace,
    trace_fallback: FuseTrace,
    variants: MatmulVariants,
    len: usize,
}

impl MatmulVariants {
    pub fn from_default(matmul: &FusedMatmul, _trace: &FuseTrace) -> Self {
        let selector = |selector: FusedMatmulSelector| {
            let mut matmul = matmul.clone();
            matmul.selector = selector;
            matmul
        };
        Self {
            simple_unit: selector(FusedMatmulSelector::SimpleUnit),
            simple_vec_mat: selector(FusedMatmulSelector::SimpleVecMat),
            double_vec_mat: selector(FusedMatmulSelector::DoubleVecMat),
            double_unit: selector(FusedMatmulSelector::DoubleUnit),
            simple: selector(FusedMatmulSelector::Simple),
            simple_multi_rows: selector(FusedMatmulSelector::SimpleMultiRows),
            double_buffering: selector(FusedMatmulSelector::DoubleBuffering),
            specialized: selector(FusedMatmulSelector::Specialized),
            ordered: selector(FusedMatmulSelector::OrderedDoubleBuffering),
        }
    }
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
    pub(crate) fn execute_fused<BT: CubeElement, S: MatmulVariantSelection>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> Result<TuneOutput<R>, TraceError<FusedMatmulError>> {
        self.info.trace.run::<R, BT, FusedMatmul>(
            &self.info.client,
            &self.info.device,
            context,
            S::select(&self.info.variants),
        )
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
        let output = TuneOutput::UnChecked(core::marker::PhantomData::<R>);

        #[cfg(feature = "autotune-checks")]
        if let TuneOutput::Checked { handles } = &mut output {
            let out_desc = context
                .tensors
                .get(&self.info.variants.simple.op.out.id)
                .unwrap();
            let handle_out = context
                .handles
                .get_handle(&out_desc.id, &burn_ir::TensorStatus::ReadOnly);

            handles.insert(
                self.info.variants.simple.op.out.id,
                (out_desc.shape.dims.clone(), handle_out.clone()),
            );
        }

        let output_write = self
            .info
            .trace_fallback
            .run::<R, BT, ElemwiseRunner>(
                &self.info.client,
                &self.info.device,
                context,
                &ElemwiseRunner,
            )
            .unwrap();

        output.merge(output_write)
    }
}

impl<R: Runtime> MatmulOptimization<R> {
    pub fn new(
        trace: FuseTrace,
        trace_fallback: FuseTrace,
        client: ComputeClient<R::Server>,
        device: R::Device,
        len: usize,
        matmul: FusedMatmul,
    ) -> Self {
        let variants = MatmulVariants::from_default(&matmul, &trace);

        let info = MatmulOptimizationInfo {
            trace,
            trace_fallback,
            client,
            device,
            len,
            variants,
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
        if arg.execute_fused::<BT, Simple>(context).is_err() {
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
            variants: state.variants.clone(),
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
            variants: self.info.variants.clone(),
            len: self.info.len,
        }
    }
}

#[derive(Default, Clone, Serialize, Deserialize, Debug)]
pub enum FusedMatmulSelector {
    #[default]
    Simple,
    SimpleMultiRows,
    DoubleBuffering,
    Specialized,
    OrderedDoubleBuffering,
    SimpleVecMat,
    DoubleVecMat,
    SimpleUnit,
    DoubleUnit,
}

#[derive(new, Clone, Serialize, Deserialize, Debug)]
pub struct FusedMatmul {
    lhs: MatmulArg,
    rhs: MatmulArg,
    out: Arg,
    pub(crate) op: BinaryOpIr,
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

impl<R: Runtime> Vectorization<R> for FusedMatmul {
    fn axis(&self, plan: &LaunchPlan<'_, R>) -> VectorizationAxis {
        let lhs_id = self.op.lhs.id;
        let rhs_id = self.op.rhs.id;

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

impl<R: Runtime> TraceRunner<R> for FusedMatmul {
    type Error = FusedMatmulError;

    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R::Server>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        configs: &'a [FuseBlockConfig],
    ) -> Result<(), FusedMatmulError> {
        match self.out.precision() {
            FusePrecision::F32 => self.matmul_fused::<R, f32>(client, inputs, outputs, &configs[0]),
            FusePrecision::Flex32 => {
                self.matmul_fused::<R, flex32>(client, inputs, outputs, &configs[0])
            }
            FusePrecision::F16 => self.matmul_fused::<R, f16>(client, inputs, outputs, &configs[0]),
            FusePrecision::BF16 => {
                self.matmul_fused::<R, bf16>(client, inputs, outputs, &configs[0])
            }
            _ => panic!("Unsupported precision"),
        }
    }
}

impl FusedMatmul {
    fn matmul_fused<'a, R: Runtime, EG: MatmulPrecision>(
        &'a self,
        client: &'a ComputeClient<R::Server>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        config: &'a FuseBlockConfig,
    ) -> Result<(), FusedMatmulError> {
        let lhs_shape = inputs.shape(self.lhs.data());
        let rhs_shape = inputs.shape(self.rhs.data());
        let out_shape = outputs.shape_ref(&config.ref_layout, config.rank as usize);

        let lhs_strides = inputs.strides(self.lhs.data());
        let rhs_strides = inputs.strides(self.rhs.data());

        let check_layout = |strides| match matrix_batch_layout(strides) {
            MatrixBatchLayout::Contiguous => (false, false),
            MatrixBatchLayout::MildlyPermuted {
                transposed,
                batch_swap: _,
            } => (false, transposed),
            MatrixBatchLayout::HighlyPermuted => (true, false),
        };

        let (lhs_make_contiguous, lhs_transposed) = check_layout(&lhs_strides);
        let (rhs_make_contiguous, rhs_transposed) = check_layout(&rhs_strides);

        if lhs_make_contiguous {
            return Err(FusedMatmulError::InvalidInput(
                "Lhs needs to be contiguous, but can't when fusing.",
            ));
        }
        if rhs_make_contiguous {
            return Err(FusedMatmulError::InvalidInput(
                "Rhs needs to be contiguous, but can't when fusing.",
            ));
        }

        let rank = lhs_shape.len();

        let m = lhs_shape[rank - 2] as u32;
        let k = lhs_shape[rank - 1] as u32;
        let n = rhs_shape[rank - 1] as u32;

        let mut line_sizes = MatmulLineSizes {
            lhs: inputs.line_size(self.lhs.data()),
            rhs: inputs.line_size(self.rhs.data()),
            out: match &config.ref_layout {
                RefLayout::Concrete(arg) => match arg {
                    Arg::Input(..) => inputs.line_size(arg),
                    Arg::Output(..) => outputs.line_size(arg),
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

        if let MatmulArg::Quantized { scheme, .. } = self.lhs {
            line_sizes.lhs *= scheme.num_quants() as u8;
        }
        if let MatmulArg::Quantized { scheme, .. } = self.rhs {
            line_sizes.rhs *= scheme.num_quants() as u8;
        }

        let problem = MatmulProblem {
            m: m as usize,
            n: n as usize,
            k: k as usize,
            lhs_batches: lhs_shape[..lhs_shape.len() - 2].to_vec(),
            rhs_batches: rhs_shape[..rhs_shape.len() - 2].to_vec(),
            out_batches: out_shape[..out_shape.len() - 2].to_vec(),
            lhs_layout: match lhs_transposed {
                true => components::MatrixLayout::ColMajor,
                false => components::MatrixLayout::RowMajor,
            },
            rhs_layout: match rhs_transposed {
                true => components::MatrixLayout::ColMajor,
                false => components::MatrixLayout::RowMajor,
            },
        };

        match &self.selector {
            FusedMatmulSelector::Simple | FusedMatmulSelector::SimpleMultiRows => {
                let multi_rows = matches!(self.selector, FusedMatmulSelector::SimpleMultiRows);

                match launch_inner_fix_dtype::<R, EG, SimpleAlgorithm<AcceleratedMatmul<Filled>>>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.lhs.clone(),
                        self.rhs.clone(),
                        Option::None,
                        self.out.clone(),
                    ),
                    outputs,
                    problem,
                    line_sizes,
                    &Selection::Inferred(SimpleArgs { multi_rows }),
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }
            FusedMatmulSelector::DoubleBuffering | FusedMatmulSelector::Specialized => {
                let specialized = matches!(self.selector, FusedMatmulSelector::Specialized);

                match launch_inner_fix_dtype::<
                    R,
                    EG,
                    CyclicDoubleBufferingAlgorithm<AcceleratedMatmul<Filled>>,
                >(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.lhs.clone(),
                        self.rhs.clone(),
                        Option::None,
                        self.out.clone(),
                    ),
                    outputs,
                    problem,
                    line_sizes,
                    &Selection::Inferred(DoubleBufferingArgs { specialized }),
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }
            FusedMatmulSelector::OrderedDoubleBuffering => {
                let row_count = match self.lhs.precision() {
                    FusePrecision::F16 | FusePrecision::BF16 => 8,
                    _ => 4,
                };

                match launch_inner_fix_dtype::<
                    R,
                    EG,
                    OrderedDoubleBufferingAlgorithm<AcceleratedMatmul<Filled>>,
                >(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.lhs.clone(),
                        self.rhs.clone(),
                        Option::None,
                        self.out.clone(),
                    ),
                    outputs,
                    problem,
                    line_sizes,
                    &Selection::Inferred(OrderedSelectionArgs {
                        row_count: Some(row_count),
                        rows_per_plane: Some(2),
                        partition_k: Some(2),
                    }),
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }
            FusedMatmulSelector::SimpleUnit => {
                match launch_inner_fix_dtype::<R, EG, SimpleUnitAlgorithm>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.lhs.clone(),
                        self.rhs.clone(),
                        Option::None,
                        self.out.clone(),
                    ),
                    outputs,
                    problem,
                    line_sizes,
                    &Default::default(),
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }
            FusedMatmulSelector::DoubleUnit => {
                match launch_inner_fix_dtype::<R, EG, DoubleUnitAlgorithm>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.lhs.clone(),
                        self.rhs.clone(),
                        Option::None,
                        self.out.clone(),
                    ),
                    outputs,
                    problem,
                    line_sizes,
                    &Default::default(),
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }
            FusedMatmulSelector::SimpleVecMat => {
                match launch_inner_fix_dtype::<R, EG, SimpleVecMatAlgorithm>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.lhs.clone(),
                        self.rhs.clone(),
                        Option::None,
                        self.out.clone(),
                    ),
                    outputs,
                    problem,
                    line_sizes,
                    &Default::default(),
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }
            FusedMatmulSelector::DoubleVecMat => {
                match launch_inner_fix_dtype::<R, EG, DoubleVecMatAlgorithm>(
                    client,
                    FusedMatmulInputLaunch::new(
                        inputs,
                        config.clone(),
                        self.lhs.clone(),
                        self.rhs.clone(),
                        Option::None,
                        self.out.clone(),
                    ),
                    outputs,
                    problem,
                    line_sizes,
                    &Default::default(),
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }
        }
    }
}

fn launch_inner_fix_dtype<'a, R: Runtime, MP: MatmulPrecision, A: Algorithm>(
    client: &ComputeClient<R::Server>,
    input: FusedMatmulInputLaunch<'a, R>,
    output: GlobalArgsLaunch<'a, R>,
    problem: MatmulProblem,
    line_sizes: MatmulLineSizes,
    selection: &Selection<A::SelectionArgs>,
) -> Result<(), MatmulSetupError> {
    let fix_plane_dim = |plane_dim: u32| {
        // Sometimes the GPU doesn't support plane instructions and doesn't report the
        // plane size, but we can still execute algorithms that don't use plane instructions.
        //
        // In this case, we set a plane size for the selector to work, defaulting to 32 as it
        // is a common plane size.
        if plane_dim == 0 { 32 } else { plane_dim }
    };

    let plane_size = fix_plane_dim(A::select_plane_dim::<R>(client));

    if <A::TileMatmul as TileMatmulFamily>::requires_accelerator()
        && tf32::supported_uses(client).contains(TypeUsage::Conversion)
    {
        match (
            TypeId::of::<LhsG<MP>>() == TypeId::of::<f32>(),
            TypeId::of::<RhsG<MP>>() == TypeId::of::<f32>(),
        ) {
            (true, true) => launch_kernel_virtual::<
                FusedMatmulSpec<(f32, f32, AccG<MP>, tf32, tf32, AccS<MP>)>,
                R,
                A,
            >(
                client, input, output, problem, line_sizes, plane_size, selection,
            ),
            (true, false) => launch_kernel_virtual::<
                FusedMatmulSpec<(f32, RhsG<MP>, AccG<MP>, tf32, RhsS<MP>, AccS<MP>)>,
                R,
                A,
            >(
                client, input, output, problem, line_sizes, plane_size, selection,
            ),
            (false, true) => launch_kernel_virtual::<
                FusedMatmulSpec<(LhsG<MP>, f32, AccG<MP>, LhsS<MP>, tf32, AccS<MP>)>,
                R,
                A,
            >(
                client, input, output, problem, line_sizes, plane_size, selection,
            ),
            (false, false) => launch_kernel_virtual::<FusedMatmulSpec<MP>, R, A>(
                client, input, output, problem, line_sizes, plane_size, selection,
            ),
        }
    } else {
        launch_kernel_virtual::<FusedMatmulSpec<MP>, R, A>(
            client, input, output, problem, line_sizes, plane_size, selection,
        )
    }
}

pub(crate) trait MatmulVariantSelection {
    fn select(variants: &MatmulVariants) -> &FusedMatmul;
}

pub(crate) struct Simple;
pub(crate) struct SimpleUnit;
pub(crate) struct SimpleVecMat;
pub(crate) struct DoubleVecMat;
pub(crate) struct DoubleUnit;
pub(crate) struct SimpleMultiRows;
pub(crate) struct DoubleBuffering;
pub(crate) struct Specialized;
pub(crate) struct Ordered;

impl MatmulVariantSelection for Simple {
    fn select(variants: &MatmulVariants) -> &FusedMatmul {
        &variants.simple
    }
}

impl MatmulVariantSelection for SimpleUnit {
    fn select(variants: &MatmulVariants) -> &FusedMatmul {
        &variants.simple_unit
    }
}

impl MatmulVariantSelection for SimpleVecMat {
    fn select(variants: &MatmulVariants) -> &FusedMatmul {
        &variants.simple_vec_mat
    }
}

impl MatmulVariantSelection for DoubleVecMat {
    fn select(variants: &MatmulVariants) -> &FusedMatmul {
        &variants.double_vec_mat
    }
}

impl MatmulVariantSelection for DoubleUnit {
    fn select(variants: &MatmulVariants) -> &FusedMatmul {
        &variants.double_unit
    }
}

impl MatmulVariantSelection for SimpleMultiRows {
    fn select(variants: &MatmulVariants) -> &FusedMatmul {
        &variants.simple_multi_rows
    }
}

impl MatmulVariantSelection for DoubleBuffering {
    fn select(variants: &MatmulVariants) -> &FusedMatmul {
        &variants.double_buffering
    }
}

impl MatmulVariantSelection for Specialized {
    fn select(variants: &MatmulVariants) -> &FusedMatmul {
        &variants.specialized
    }
}

impl MatmulVariantSelection for Ordered {
    fn select(variants: &MatmulVariants) -> &FusedMatmul {
        &variants.ordered
    }
}
