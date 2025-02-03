use std::any::TypeId;

use crate::fusion::elemwise::optimization::ElemwiseRunner;
use crate::fusion::on_write::ir::ElemwisePrecision;
use crate::kernel::matmul;
use crate::{fusion::JitFusionHandle, JitRuntime};
use crate::{BoolElement, FloatElement};

use burn_fusion::stream::Context;
use burn_tensor::repr::{BinaryOperationDescription, TensorStatus};
use burn_tensor::Shape;
use cubecl::linalg::matmul::components;
use cubecl::linalg::matmul::components::tile::accelerated::Accelerated;
use cubecl::linalg::matmul::components::MatmulProblem;
use cubecl::linalg::matmul::kernels::matmul::{
    MatmulSelector, PipelinedSelector, SpecializedSelector, StandardSelector,
};
use cubecl::linalg::matmul::kernels::{MatmulAvailabilityError, MatmulLaunchError};
use cubecl::linalg::tensor::{matrix_layout, MatrixLayout};
use cubecl::{client::ComputeClient, prelude::*};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

use crate::fusion::on_write::{
    ir::{Arg, ElemwiseConfig, GlobalArgsLaunch},
    trace::{FuseOnWriteTrace, TraceRunner},
};

use super::args::FusedMatmulInputLaunch;
use super::spec::FusedMatmulSpec;
use super::tune::fused_matmul_autotune;

/// Fuse matmul operation followed by elemwise operations into a single kernel.
pub struct MatmulOptimization<R: JitRuntime> {
    trace: FuseOnWriteTrace,
    trace_fallback: FuseOnWriteTrace,
    pub(crate) client: ComputeClient<R::Server, R::Channel>,
    pub(crate) device: R::Device,
    pub(crate) len: usize,
    pub(crate) matmul_standard: FusedMatmul,
    pub(crate) matmul_pipelined: FusedMatmul,
    pub(crate) matmul_specialized: FusedMatmul,
}

#[derive(Serialize, Deserialize, Debug)]
/// State for the [matrix optimization](MatmulOptimizationState).
pub struct MatmulOptimizationState {
    trace: FuseOnWriteTrace,
    trace_fallback: FuseOnWriteTrace,
    matmul_standard: FusedMatmul,
    matmul_pipelined: FusedMatmul,
    matmul_specialized: FusedMatmul,
    len: usize,
}

impl<R: JitRuntime> MatmulOptimization<R> {
    pub fn new(
        trace: FuseOnWriteTrace,
        trace_fallback: FuseOnWriteTrace,
        client: ComputeClient<R::Server, R::Channel>,
        device: R::Device,
        len: usize,
        matmul: FusedMatmul,
    ) -> Self {
        let mut matmul_standard = matmul.clone();
        let mut matmul_specialized = matmul.clone();
        let mut matmul_pipelined = matmul;

        matmul_standard.selector = FusedMatmulSelector::Standard;
        matmul_specialized.selector = FusedMatmulSelector::Specialized;
        matmul_pipelined.selector = FusedMatmulSelector::Pipelined;

        Self {
            trace,
            trace_fallback,
            client,
            device,
            len,
            matmul_standard,
            matmul_pipelined,
            matmul_specialized,
        }
    }
    /// Execute the optimization.
    pub fn execute<BT: BoolElement>(&mut self, context: &mut Context<'_, JitFusionHandle<R>>) {
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
    pub fn from_state(device: &R::Device, state: MatmulOptimizationState) -> Self {
        Self {
            trace: state.trace,
            trace_fallback: state.trace_fallback,
            len: state.len,
            client: R::client(device),
            device: device.clone(),
            matmul_standard: state.matmul_standard.clone(),
            matmul_specialized: state.matmul_specialized.clone(),
            matmul_pipelined: state.matmul_pipelined.clone(),
        }
    }

    /// Convert the optimization to its [state](MatmulOptimizationState).
    pub fn to_state(&self) -> MatmulOptimizationState {
        MatmulOptimizationState {
            trace: self.trace.clone(),
            trace_fallback: self.trace_fallback.clone(),
            matmul_standard: self.matmul_standard.clone(),
            matmul_specialized: self.matmul_specialized.clone(),
            matmul_pipelined: self.matmul_pipelined.clone(),
            len: self.len,
        }
    }

    pub fn execute_standard_fused<BT: BoolElement>(
        &self,
        context: &mut Context<'_, JitFusionHandle<R>>,
    ) -> Result<(), FusedMatmulError> {
        self.trace.run::<R, BT, FusedMatmul>(
            &self.client,
            &self.device,
            context,
            &self.matmul_standard,
        )
    }

    pub fn execute_specialized_fused<BT: BoolElement>(
        &self,
        context: &mut Context<'_, JitFusionHandle<R>>,
    ) -> Result<(), FusedMatmulError> {
        self.trace.run::<R, BT, FusedMatmul>(
            &self.client,
            &self.device,
            context,
            &self.matmul_specialized,
        )
    }

    pub fn execute_pipelined_fused<BT: BoolElement>(
        &self,
        context: &mut Context<'_, JitFusionHandle<R>>,
    ) -> Result<(), FusedMatmulError> {
        self.trace.run::<R, BT, FusedMatmul>(
            &self.client,
            &self.device,
            context,
            &self.matmul_pipelined,
        )
    }

    pub fn execute_fallback<BT: BoolElement>(&self, context: &mut Context<'_, JitFusionHandle<R>>) {
        match self.matmul_standard.lhs.precision() {
            ElemwisePrecision::F32 => self.run_fallback::<BT, f32>(context),
            ElemwisePrecision::F16 => self.run_fallback::<BT, f16>(context),
            ElemwisePrecision::BF16 => self.run_fallback::<BT, bf16>(context),
            _ => panic!("Unsupported precision"),
        }
    }

    fn run_fallback<BT: BoolElement, EG: FloatElement>(
        &self,
        context: &mut Context<'_, JitFusionHandle<R>>,
    ) {
        let (out_tensor, out_desc) = {
            let lhs = context
                .tensors
                .get(&self.matmul_standard.op.lhs.id)
                .unwrap()
                .clone();
            let rhs = context
                .tensors
                .get(&self.matmul_standard.op.rhs.id)
                .unwrap()
                .clone();
            let out = context
                .tensors
                .get(&self.matmul_standard.op.out.id)
                .unwrap()
                .clone();

            let lhs_handle = context.handles.get_handle(&lhs.id, &TensorStatus::ReadOnly);
            let rhs_handle = context.handles.get_handle(&rhs.id, &TensorStatus::ReadOnly);

            let lhs_tensor = lhs_handle.into_tensor(Shape {
                dims: lhs.shape.clone(),
            });
            let rhs_tensor = rhs_handle.into_tensor(Shape {
                dims: rhs.shape.clone(),
            });
            let out_tensor = matmul::matmul::<R, EG>(
                lhs_tensor,
                rhs_tensor,
                None,
                matmul::MatmulStrategy::default(),
            )
            .unwrap();
            (out_tensor, out)
        };
        context
            .handles
            .register_handle(out_desc.id, JitFusionHandle::from(out_tensor));

        self.trace_fallback
            .run::<R, BT, ElemwiseRunner>(&self.client, &self.device, context, &ElemwiseRunner)
            .unwrap();
    }
}

#[derive(Default, Clone, Serialize, Deserialize, Debug)]
pub enum FusedMatmulSelector {
    #[default]
    Standard,
    Pipelined,
    Specialized,
}

#[derive(new, Clone, Serialize, Deserialize, Debug)]
pub struct FusedMatmul {
    lhs: Arg,
    rhs: Arg,
    out: Arg,
    pub(crate) op: BinaryOperationDescription,
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

impl<R: JitRuntime> TraceRunner<R> for FusedMatmul {
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
    fn matmul_fused<'a, R: JitRuntime, EG: Numeric>(
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
            FusedMatmulSelector::Standard => {
                match matmul_launch_kernel::<R, EG, StandardSelector<Accelerated>>(
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
            FusedMatmulSelector::Pipelined => {
                match matmul_launch_kernel::<R, EG, PipelinedSelector<Accelerated>>(
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
                match matmul_launch_kernel::<R, EG, SpecializedSelector<Accelerated>>(
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

fn matmul_launch_kernel<'a, R: Runtime, EG: Numeric, S: MatmulSelector>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: FusedMatmulInputLaunch<'a, R>,
    output: GlobalArgsLaunch<'a, R>,
    problem: MatmulProblem,
    plane_size: u32,
) -> Result<(), MatmulLaunchError> {
    if TypeId::of::<EG>() == TypeId::of::<half::f16>()
        || TypeId::of::<EG>() == TypeId::of::<flex32>()
    {
        S::select_kernel::<FusedMatmulSpec<EG, half::f16, f32>, R>(
            client, input, output, problem, plane_size,
        )
    } else if TypeId::of::<EG>() == TypeId::of::<half::bf16>() {
        S::select_kernel::<FusedMatmulSpec<EG, half::bf16, f32>, R>(
            client, input, output, problem, plane_size,
        )
    } else if S::stage_tf32_supported() {
        S::select_kernel::<FusedMatmulSpec<EG, tf32, f32>, R>(
            client, input, output, problem, plane_size,
        )
    } else {
        S::select_kernel::<FusedMatmulSpec<EG, EG, f32>, R>(
            client, input, output, problem, plane_size,
        )
    }
}
