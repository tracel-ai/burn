use std::any::TypeId;

use crate::fusion::on_write::ir::ElemwisePrecision;
use crate::BoolElement;
use crate::{fusion::JitFusionHandle, JitRuntime};
use burn_fusion::stream::Context;
use burn_tensor::repr::TensorDescription;
use cubecl::linalg::matmul;
use cubecl::linalg::matmul::components::MatmulProblem;
use cubecl::linalg::matmul::kernels::matmul::{CmmaSelector, PlaneMmaSelector};
use cubecl::linalg::matmul::kernels::{MatmulAvailabilityError, MatmulLaunchError};
use cubecl::{client::ComputeClient, prelude::*};
use serde::{Deserialize, Serialize};

use crate::fusion::on_write::{
    ir::{Arg, ElemwiseConfig, GlobalArgsLaunch},
    trace::{FuseOnWriteTrace, TraceRunner},
};

use super::spec::FusedMatmulSpec;
use super::FusedMatmulInputLaunch;

#[derive(new)]
/// Fuse element wise operations into a single kernel.
pub struct MatmulOptimization<R: JitRuntime> {
    trace: FuseOnWriteTrace,
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    len: usize,
    args: (Arg, Arg, Arg),
}

#[derive(Serialize, Deserialize, Debug)]
/// State for the [elemwise optimization](ElemwiseOptimization).
pub struct MatmulOptimizationState {
    trace: FuseOnWriteTrace,
    args: (Arg, Arg, Arg),
    len: usize,
}

impl<R: JitRuntime> MatmulOptimization<R> {
    /// Execute the optimization.
    pub fn execute<BT: BoolElement>(&mut self, context: &mut Context<'_, JitFusionHandle<R>>) {
        self.trace
            .run::<R, BT, Self>(&self.client, &self.device, context, self)
    }

    /// Number of element wise operations fused.
    pub fn num_ops_fused(&self) -> usize {
        self.len
    }

    /// Create an optimization from its [state](ElemwiseOptimizationState).
    pub fn from_state(device: &R::Device, state: MatmulOptimizationState) -> Self {
        Self {
            trace: state.trace,
            len: state.len,
            client: R::client(device),
            device: device.clone(),
            args: state.args.clone(),
        }
    }

    /// Convert the optimization to its [state](ElemwiseOptimizationState).
    pub fn to_state(&self) -> MatmulOptimizationState {
        MatmulOptimizationState {
            trace: self.trace.clone(),
            args: self.args.clone(),
            len: self.len,
        }
    }
}

impl<R: JitRuntime> TraceRunner<R> for MatmulOptimization<R> {
    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R::Server, R::Channel>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        config: &'a ElemwiseConfig,
    ) {
        match self.args.2.precision() {
            ElemwisePrecision::F32 => {
                matmul_cmma_no_check::<R, f32>(client, inputs, outputs, config, &self.args)
            }
            ElemwisePrecision::F16 => {
                matmul_cmma_no_check::<R, half::f16>(client, inputs, outputs, config, &self.args)
            }
            ElemwisePrecision::BF16 => {
                matmul_cmma_no_check::<R, half::bf16>(client, inputs, outputs, config, &self.args)
            }
            _ => panic!("Unsupported precision"),
        }
        .unwrap()
    }

    fn vectorization<'a>(
        handles_inputs: impl Iterator<Item = &'a JitFusionHandle<R>>,
        inputs: impl Iterator<Item = &'a TensorDescription>,
        outputs: impl Iterator<Item = &'a TensorDescription>,
    ) -> u8 {
        let vectorization_input = |handle: &JitFusionHandle<R>, desc: &TensorDescription| {
            let rank = handle.strides.len();

            // Last dimension strides should be 1, otherwise vecX won't be contiguous.
            if handle.strides[rank - 1] != 1 {
                return 1;
            }

            for s in R::max_line_size_elem(&desc.dtype.into()) {
                // The last dimension should be a multiple of the vector size.
                if desc.shape[rank - 1] % s as usize == 0 {
                    return s;
                }
            }

            1
        };

        let vectorization_output = |desc: &TensorDescription| {
            let rank = desc.shape.len();

            for s in R::max_line_size_elem(&desc.dtype.into()) {
                // The last dimension should be a multiple of the vector size.
                if desc.shape[rank - 1] % s as usize == 0 {
                    return s;
                }
            }

            1
        };

        let mut output = u8::MAX;

        for (handle, tensor) in handles_inputs.zip(inputs) {
            output = Ord::min(vectorization_input(handle, tensor), output);
        }

        for tensor in outputs {
            output = Ord::min(vectorization_output(tensor), output);
        }

        output
    }
}

fn matmul_cmma_no_check<'a, R: Runtime, EG: Numeric>(
    client: &'a ComputeClient<R::Server, R::Channel>,
    inputs: GlobalArgsLaunch<'a, R>,
    outputs: GlobalArgsLaunch<'a, R>,
    config: &'a ElemwiseConfig,
    (lhs, rhs, out): &'a (Arg, Arg, Arg),
) -> Result<(), MatmulLaunchError> {
    let transposed: (bool, bool) = (false, false);
    let disable_cmma: bool = false;

    let lhs_shape = inputs.shape(lhs);
    let rhs_shape = inputs.shape(rhs);

    let rank = lhs_shape.len();

    let m = lhs_shape[rank - 2] as u32;
    let k = lhs_shape[rank - 1] as u32;
    let n = rhs_shape[rank - 1] as u32;

    let lhs_line_size = inputs.line_size(&lhs);
    let rhs_line_size = inputs.line_size(&lhs);
    let out_line_size = outputs.line_size(&config.ref_layout);

    let problem = MatmulProblem {
        m: m as usize,
        n: n as usize,
        k: k as usize,
        batches: (
            lhs_shape[..lhs_shape.len() - 2].to_vec(),
            rhs_shape[..rhs_shape.len() - 2].to_vec(),
        ),
        lhs_layout: match transposed.0 {
            true => matmul::components::MatrixLayout::ColMajor,
            false => matmul::components::MatrixLayout::RowMajor,
        },
        rhs_layout: match transposed.1 {
            true => matmul::components::MatrixLayout::ColMajor,
            false => matmul::components::MatrixLayout::RowMajor,
        },
        lhs_line_size,
        rhs_line_size,
        out_line_size,
    };

    let plane_size = client
        .properties()
        .hardware_properties()
        .defined_plane_size();

    match plane_size {
        Some(32) => matmul_launch_kernel::<32, R, EG>(
            client,
            FusedMatmulInputLaunch::new(inputs, config, &lhs, &rhs, &out),
            outputs,
            disable_cmma,
            problem,
        ),
        Some(64) => matmul_launch_kernel::<64, R, EG>(
            client,
            FusedMatmulInputLaunch::new(inputs, config, &lhs, &rhs, &out),
            outputs,
            disable_cmma,
            problem,
        ),
        Some(plane_dim) => Err(MatmulLaunchError::Unavailable(
            MatmulAvailabilityError::PlaneDimUnsupported { plane_dim },
        )),
        None => Err(MatmulLaunchError::Unavailable(
            MatmulAvailabilityError::PlaneDimUnknown,
        )),
    }
}

fn matmul_launch_kernel<'a, const PLANE_DIM: u32, R: Runtime, EG: Numeric>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: FusedMatmulInputLaunch<'a, R>,
    output: GlobalArgsLaunch<'a, R>,
    disable_cmma: bool,
    problem: MatmulProblem,
) -> Result<(), MatmulLaunchError> {
    if disable_cmma {
        PlaneMmaSelector::select_kernel::<FusedMatmulSpec<{ PLANE_DIM }, EG, EG, f32>, R>(
            client, input, output, problem,
        )
    } else if TypeId::of::<EG>() == TypeId::of::<half::f16>()
        || TypeId::of::<EG>() == TypeId::of::<flex32>()
    {
        CmmaSelector::select_kernel::<FusedMatmulSpec<{ PLANE_DIM }, EG, half::f16, f32>, R>(
            client, input, output, problem,
        )
    } else {
        CmmaSelector::select_kernel::<FusedMatmulSpec<{ PLANE_DIM }, EG, tf32, f32>, R>(
            client, input, output, problem,
        )
    }
}
