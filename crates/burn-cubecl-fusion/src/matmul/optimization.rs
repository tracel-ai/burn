use std::any::TypeId;
use std::sync::Arc;

use crate::CubeFusionHandle;
use crate::elemwise::optimization::ElemwiseRunner;
use crate::shared::ir::FusePrecision;
use crate::shared::ir::RefLayout;
use crate::shared::trace::TraceError;
use crate::shared::trace::TuneOutput;
use crate::shared::trace::Vectorization;

use burn_fusion::stream::Context;
use burn_ir::{BinaryOpIr, TensorStatus};
use cubecl::linalg::matmul::components;
use cubecl::linalg::matmul::components::MatmulPrecision;
use cubecl::linalg::matmul::components::MatmulProblem;
use cubecl::linalg::matmul::components::tile::TileMatmulFamily;
use cubecl::linalg::matmul::components::tile::accelerated::Accelerated;
use cubecl::linalg::matmul::kernels::matmul::double_buffering::DoubleBufferingAlgorithm;
use cubecl::linalg::matmul::kernels::matmul::simple::SimpleAlgorithm;
use cubecl::linalg::matmul::kernels::matmul::{Algorithm, select_kernel};
use cubecl::linalg::matmul::kernels::{MatmulAvailabilityError, MatmulLaunchError};
use cubecl::linalg::tensor::{MatrixBatchLayout, matrix_batch_layout};
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
    trace: FuseTrace,
    trace_fallback: FuseTrace,
    pub(crate) client: ComputeClient<R::Server, R::Channel>,
    pub(crate) device: R::Device,
    pub(crate) len: usize,
    pub(crate) matmul_simple: FusedMatmul,
    pub(crate) matmul_double_buffering: FusedMatmul,
    fallback: Arc<dyn MatmulFallbackFn<R>>,
}

pub trait MatmulFallbackFn<R: Runtime>: Send + Sync {
    fn run(
        &self,
        lhs: (CubeFusionHandle<R>, &[usize]),
        rhs: (CubeFusionHandle<R>, &[usize]),
    ) -> CubeFusionHandle<R>;
}

#[derive(Serialize, Deserialize, Debug)]
/// State for the [matrix optimization](MatmulOptimizationState).
pub struct MatmulOptimizationState {
    trace: FuseTrace,
    trace_fallback: FuseTrace,
    matmul_simple: FusedMatmul,
    matmul_double_buffering: FusedMatmul,
    len: usize,
}

impl<R: Runtime> MatmulOptimization<R> {
    pub fn new(
        trace: FuseTrace,
        trace_fallback: FuseTrace,
        client: ComputeClient<R::Server, R::Channel>,
        device: R::Device,
        len: usize,
        matmul: FusedMatmul,
        fallback: Arc<dyn MatmulFallbackFn<R>>,
    ) -> Self {
        let mut matmul_simple = matmul.clone();
        let mut matmul_double_buffering = matmul;

        matmul_simple.selector = FusedMatmulSelector::Simple;
        matmul_double_buffering.selector = FusedMatmulSelector::DoubleBuffering;

        Self {
            trace,
            trace_fallback,
            client,
            device,
            len,
            matmul_simple,
            matmul_double_buffering,
            fallback,
        }
    }
    /// Execute the optimization.
    pub fn execute<BT: CubeElement>(&mut self, context: &mut Context<'_, CubeFusionHandle<R>>) {
        #[cfg(feature = "autotune")]
        fused_matmul_autotune::<R, BT>(self, context);

        #[cfg(not(feature = "autotune"))]
        if self.execute_standard_fused::<BT>(context).is_err() {
            self.execute_fallback::<BT>(context);
        }
    }

    /// Number of operations fused.
    pub fn num_ops_fused(&self) -> usize {
        self.len
    }

    /// Create an optimization from its [state](MatmulOptimizationState).
    pub fn from_state(
        device: &R::Device,
        state: MatmulOptimizationState,
        fallback: Arc<dyn MatmulFallbackFn<R>>,
    ) -> Self {
        Self {
            trace: state.trace,
            trace_fallback: state.trace_fallback,
            len: state.len,
            client: R::client(device),
            device: device.clone(),
            matmul_simple: state.matmul_simple.clone(),
            matmul_double_buffering: state.matmul_double_buffering.clone(),
            fallback,
        }
    }

    /// Convert the optimization to its [state](MatmulOptimizationState).
    pub fn to_state(&self) -> MatmulOptimizationState {
        MatmulOptimizationState {
            trace: self.trace.clone(),
            trace_fallback: self.trace_fallback.clone(),
            matmul_simple: self.matmul_simple.clone(),
            matmul_double_buffering: self.matmul_double_buffering.clone(),
            len: self.len,
        }
    }

    /// Returns the number of output buffers added by fusion.
    pub fn num_output_buffers(&self) -> usize {
        self.trace_fallback.resources.outputs.len()
    }

    pub fn execute_simple_fused<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> Result<TuneOutput<R>, TraceError<FusedMatmulError>> {
        self.trace.run::<R, BT, FusedMatmul>(
            &self.client,
            &self.device,
            context,
            &self.matmul_simple,
        )
    }

    pub fn execute_double_buffering_fused<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> Result<TuneOutput<R>, TraceError<FusedMatmulError>> {
        self.trace.run::<R, BT, FusedMatmul>(
            &self.client,
            &self.device,
            context,
            &self.matmul_double_buffering,
        )
    }

    pub fn execute_fallback<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> TuneOutput<R> {
        let (out_tensor, out_desc) = {
            let lhs = context
                .tensors
                .get(&self.matmul_simple.op.lhs.id)
                .unwrap()
                .clone();
            let rhs = context
                .tensors
                .get(&self.matmul_simple.op.rhs.id)
                .unwrap()
                .clone();
            let out = context
                .tensors
                .get(&self.matmul_simple.op.out.id)
                .unwrap()
                .clone();

            let lhs_handle = context.handles.get_handle(&lhs.id, &TensorStatus::ReadOnly);
            let rhs_handle = context.handles.get_handle(&rhs.id, &TensorStatus::ReadOnly);
            let out_handle = self
                .fallback
                .run((lhs_handle, &lhs.shape), (rhs_handle, &rhs.shape));

            (out_handle, out)
        };
        #[cfg(feature = "autotune-checks")]
        let mut output = TuneOutput::Checked {
            handles: Default::default(),
        };
        #[cfg(not(feature = "autotune-checks"))]
        let output = TuneOutput::UnChecked(core::marker::PhantomData::<R>);

        #[cfg(feature = "autotune-checks")]
        if let TuneOutput::Checked { handles } = &mut output {
            handles.insert(
                self.matmul_simple.op.out.id,
                (out_desc.shape.clone(), out_tensor.clone()),
            );
        }
        context.handles.register_handle(out_desc.id, out_tensor);

        let output_write = self
            .trace_fallback
            .run::<R, BT, ElemwiseRunner>(&self.client, &self.device, context, &ElemwiseRunner)
            .unwrap();

        output.merge(output_write)
    }
}

#[derive(Default, Clone, Serialize, Deserialize, Debug)]
pub enum FusedMatmulSelector {
    #[default]
    Simple,
    DoubleBuffering,
}

#[derive(new, Clone, Serialize, Deserialize, Debug)]
pub struct FusedMatmul {
    lhs: Arg,
    rhs: Arg,
    out: Arg,
    pub(crate) op: BinaryOpIr,
    pub(crate) selector: FusedMatmulSelector,
}

#[derive(Debug)]
pub enum FusedMatmulError {
    LaunchError(MatmulLaunchError),
    InvalidInput,
}

impl From<MatmulLaunchError> for FusedMatmulError {
    fn from(value: MatmulLaunchError) -> Self {
        Self::LaunchError(value)
    }
}

impl<R: Runtime> Vectorization<R> for FusedMatmul {}

impl<R: Runtime> TraceRunner<R> for FusedMatmul {
    type Error = FusedMatmulError;

    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R::Server, R::Channel>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        configs: &'a [FuseBlockConfig],
    ) -> Result<(), FusedMatmulError> {
        match self.out.precision() {
            FusePrecision::F32 => self.matmul_fused::<R, f32>(client, inputs, outputs, &configs[0]),
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
        client: &'a ComputeClient<R::Server, R::Channel>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        config: &'a FuseBlockConfig,
    ) -> Result<(), FusedMatmulError> {
        let lhs_shape = inputs.shape(&self.lhs);
        let rhs_shape = inputs.shape(&self.rhs);

        let lhs_strides = inputs.strides(&self.lhs);
        let rhs_strides = inputs.strides(&self.rhs);

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

        if lhs_make_contiguous || rhs_make_contiguous {
            return Err(FusedMatmulError::InvalidInput);
        }

        let rank = lhs_shape.len();

        let m = lhs_shape[rank - 2] as u32;
        let k = lhs_shape[rank - 1] as u32;
        let n = rhs_shape[rank - 1] as u32;

        let lhs_line_size = inputs.line_size(&self.lhs);
        let rhs_line_size = inputs.line_size(&self.rhs);
        let out_line_size = match &config.ref_layout {
            RefLayout::Concrete(arg) => match arg {
                Arg::Input(..) => inputs.line_size(arg),
                Arg::Output(..) => outputs.line_size(arg),
                _ => panic!("Invalid ref layout"),
            },
            RefLayout::Virtual(_) => 1,
        };

        if out_line_size == 1 && (lhs_line_size > 1 || rhs_line_size > 1) {
            return Err(FusedMatmulError::InvalidInput);
        }

        let problem = MatmulProblem {
            m: m as usize,
            n: n as usize,
            k: k as usize,
            batches: (
                lhs_shape[..lhs_shape.len() - 2].to_vec(),
                rhs_shape[..rhs_shape.len() - 2].to_vec(),
            ),
            lhs_layout: match lhs_transposed {
                true => components::MatrixLayout::ColMajor,
                false => components::MatrixLayout::RowMajor,
            },
            rhs_layout: match rhs_transposed {
                true => components::MatrixLayout::ColMajor,
                false => components::MatrixLayout::RowMajor,
            },
            lhs_line_size,
            rhs_line_size,
            out_line_size,
        };

        let plane_size = client
            .properties()
            .hardware_properties()
            .defined_plane_size();

        let plane_size = match plane_size {
            Some(val) => val,
            None => {
                return Err(MatmulLaunchError::Unavailable(
                    MatmulAvailabilityError::PlaneDimUnknown,
                )
                .into());
            }
        };

        match self.selector {
            FusedMatmulSelector::Simple => {
                match matmul_launch_kernel::<R, EG, SimpleAlgorithm<Accelerated>>(
                    client,
                    FusedMatmulInputLaunch::new(inputs, config, &self.lhs, &self.rhs, &self.out),
                    outputs,
                    problem,
                    plane_size,
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }
            FusedMatmulSelector::DoubleBuffering => {
                match matmul_launch_kernel::<R, EG, DoubleBufferingAlgorithm<Accelerated>>(
                    client,
                    FusedMatmulInputLaunch::new(inputs, config, &self.lhs, &self.rhs, &self.out),
                    outputs,
                    problem,
                    plane_size,
                ) {
                    Ok(_) => Ok(()),
                    Err(err) => Err(FusedMatmulError::LaunchError(err)),
                }
            }
        }
    }
}

fn matmul_launch_kernel<'a, R: Runtime, EG: MatmulPrecision, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: FusedMatmulInputLaunch<'a, R>,
    output: GlobalArgsLaunch<'a, R>,
    problem: MatmulProblem,
    plane_size: u32,
) -> Result<(), MatmulLaunchError> {
    if <A::TileMatmul as TileMatmulFamily>::requires_tensor_cores()
        && TypeId::of::<EG>() == TypeId::of::<f32>()
    {
        select_kernel::<FusedMatmulSpec<(f32, tf32, f32, f32)>, R, A>(
            client, input, output, problem, plane_size,
        )
    } else {
        select_kernel::<FusedMatmulSpec<EG>, R, A>(client, input, output, problem, plane_size)
    }
}
