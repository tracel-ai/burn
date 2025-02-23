use std::any::TypeId;
use std::sync::Arc;

use crate::elemwise::optimization::ElemwiseRunner;
use crate::on_write::ir::ElemwisePrecision;
use crate::CubeFusionHandle;

use burn_fusion::stream::Context;
use burn_ir::{BinaryOpIr, TensorStatus};
use cubecl::linalg::matmul::components;
use cubecl::linalg::matmul::components::tile::accelerated::Accelerated;
use cubecl::linalg::matmul::components::tile::TileMatmulFamily;
use cubecl::linalg::matmul::components::MatmulProblem;
use cubecl::linalg::matmul::kernels::matmul::double_buffering::DoubleBufferingAlgorithm;
use cubecl::linalg::matmul::kernels::matmul::simple::SimpleAlgorithm;
use cubecl::linalg::matmul::kernels::matmul::specialized::SpecializedAlgorithm;
use cubecl::linalg::matmul::kernels::matmul::{select_kernel, Algorithm};
use cubecl::linalg::matmul::kernels::{MatmulAvailabilityError, MatmulLaunchError};
use cubecl::linalg::tensor::{matrix_layout, MatrixLayout};
use cubecl::{client::ComputeClient, prelude::*};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

use crate::on_write::{
    ir::{Arg, ElemwiseConfig, GlobalArgsLaunch},
    trace::{FuseOnWriteTrace, TraceRunner},
};

use super::args::FusedMatmulInputLaunch;
use super::spec::FusedMatmulSpec;
use super::tune::fused_matmul_autotune;

/// Fuse matmul operation followed by elemwise operations into a single kernel.
pub struct MatmulOptimization<R: Runtime> {
    trace: FuseOnWriteTrace,
    trace_fallback: FuseOnWriteTrace,
    pub(crate) client: ComputeClient<R::Server, R::Channel>,
    pub(crate) device: R::Device,
    pub(crate) len: usize,
    pub(crate) matmul_simple: FusedMatmul,
    pub(crate) matmul_double_buffering: FusedMatmul,
    pub(crate) matmul_specialized: FusedMatmul,
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
    trace: FuseOnWriteTrace,
    trace_fallback: FuseOnWriteTrace,
    matmul_simple: FusedMatmul,
    matmul_double_buffering: FusedMatmul,
    matmul_specialized: FusedMatmul,
    len: usize,
}

impl<R: Runtime> MatmulOptimization<R> {
    pub fn new(
        trace: FuseOnWriteTrace,
        trace_fallback: FuseOnWriteTrace,
        client: ComputeClient<R::Server, R::Channel>,
        device: R::Device,
        len: usize,
        matmul: FusedMatmul,
        fallback: Arc<dyn MatmulFallbackFn<R>>,
    ) -> Self {
        let mut matmul_simple = matmul.clone();
        let mut matmul_specialized = matmul.clone();
        let mut matmul_double_buffering = matmul;

        matmul_simple.selector = FusedMatmulSelector::Simple;
        matmul_specialized.selector = FusedMatmulSelector::Specialized;
        matmul_double_buffering.selector = FusedMatmulSelector::DoubleBuffering;

        Self {
            trace,
            trace_fallback,
            client,
            device,
            len,
            matmul_simple,
            matmul_double_buffering,
            matmul_specialized,
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
            matmul_specialized: state.matmul_specialized.clone(),
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
            matmul_specialized: self.matmul_specialized.clone(),
            matmul_double_buffering: self.matmul_double_buffering.clone(),
            len: self.len,
        }
    }

    /// Returns the number of output buffers added by fusion.
    pub fn num_output_buffers(&self) -> usize {
        self.trace_fallback.outputs.len()
    }

    pub fn execute_simple_fused<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> Result<(), FusedMatmulError> {
        self.trace.run::<R, BT, FusedMatmul>(
            &self.client,
            &self.device,
            context,
            &self.matmul_simple,
        )
    }

    pub fn execute_specialized_fused<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> Result<(), FusedMatmulError> {
        self.trace.run::<R, BT, FusedMatmul>(
            &self.client,
            &self.device,
            context,
            &self.matmul_specialized,
        )
    }

    pub fn execute_double_buffering_fused<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> Result<(), FusedMatmulError> {
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
    ) {
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
        context.handles.register_handle(out_desc.id, out_tensor);

        self.trace_fallback
            .run::<R, BT, ElemwiseRunner>(&self.client, &self.device, context, &ElemwiseRunner)
            .unwrap();
    }
}

#[derive(Default, Clone, Serialize, Deserialize, Debug)]
pub enum FusedMatmulSelector {
    #[default]
    Simple,
    DoubleBuffering,
    Specialized,
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

impl<R: Runtime> TraceRunner<R> for FusedMatmul {
    type Error = FusedMatmulError;

    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R::Server, R::Channel>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        config: &'a ElemwiseConfig,
    ) -> Result<(), FusedMatmulError> {
        match self.out.precision() {
            ElemwisePrecision::F32 => self.matmul_fused::<R, f32>(client, inputs, outputs, config),
            ElemwisePrecision::F16 => self.matmul_fused::<R, f16>(client, inputs, outputs, config),
            ElemwisePrecision::BF16 => {
                self.matmul_fused::<R, bf16>(client, inputs, outputs, config)
            }
            _ => panic!("Unsupported precision"),
        }
    }
}

impl FusedMatmul {
    fn matmul_fused<'a, R: Runtime, EG: Numeric>(
        &'a self,
        client: &'a ComputeClient<R::Server, R::Channel>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        config: &'a ElemwiseConfig,
    ) -> Result<(), FusedMatmulError> {
        let lhs_shape = inputs.shape(&self.lhs);
        let rhs_shape = inputs.shape(&self.rhs);

        let lhs_strides = inputs.strides(&self.lhs);
        let rhs_strides = inputs.strides(&self.rhs);

        let check_layout = |strides| match matrix_layout(strides) {
            MatrixLayout::Contiguous => (false, false),
            MatrixLayout::MildlyPermuted {
                transposed,
                batch_swap: _,
            } => (false, transposed),
            MatrixLayout::HighlyPermuted => (true, false),
        };

        let (lhs_make_contiguous, lhs_transposed) = check_layout(lhs_strides);
        let (rhs_make_contiguous, rhs_transposed) = check_layout(rhs_strides);

        if lhs_make_contiguous || rhs_make_contiguous {
            return Err(FusedMatmulError::InvalidInput);
        }

        let rank = lhs_shape.len();

        let m = lhs_shape[rank - 2] as u32;
        let k = lhs_shape[rank - 1] as u32;
        let n = rhs_shape[rank - 1] as u32;

        let lhs_line_size = inputs.line_size(&self.lhs);
        let rhs_line_size = inputs.line_size(&self.rhs);
        let out_line_size = match config.ref_layout {
            Arg::Input(..) => inputs.line_size(&config.ref_layout),
            Arg::Output(..) => outputs.line_size(&config.ref_layout),
            _ => panic!("Invalid ref layout"),
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
                .into())
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
            FusedMatmulSelector::Specialized => {
                match matmul_launch_kernel::<R, EG, SpecializedAlgorithm<Accelerated>>(
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

fn matmul_launch_kernel<'a, R: Runtime, EG: Numeric, A: Algorithm>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: FusedMatmulInputLaunch<'a, R>,
    output: GlobalArgsLaunch<'a, R>,
    problem: MatmulProblem,
    plane_size: u32,
) -> Result<(), MatmulLaunchError> {
    if TypeId::of::<EG>() == TypeId::of::<half::f16>()
        || TypeId::of::<EG>() == TypeId::of::<flex32>()
    {
        select_kernel::<FusedMatmulSpec<EG, half::f16, f32>, R, A>(
            client, input, output, problem, plane_size, false,
        )
    } else if TypeId::of::<EG>() == TypeId::of::<half::bf16>() {
        select_kernel::<FusedMatmulSpec<EG, half::bf16, f32>, R, A>(
            client, input, output, problem, plane_size, false,
        )
    } else if <A::TileMatmul as TileMatmulFamily>::requires_tensor_cores() {
        select_kernel::<FusedMatmulSpec<EG, tf32, f32>, R, A>(
            client, input, output, problem, plane_size, false,
        )
    } else {
        select_kernel::<FusedMatmulSpec<EG, EG, f32>, R, A>(
            client, input, output, problem, plane_size, false,
        )
    }
}
