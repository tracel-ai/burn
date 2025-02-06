use crate::{fusion::on_write::kernel::fuse_on_write, BoolElement};
use crate::{fusion::JitFusionHandle, JitRuntime};
use burn_fusion::stream::Context;
use cubecl::{calculate_cube_count_elemwise, client::ComputeClient, prelude::*, CubeDim};
use serde::{Deserialize, Serialize};

use crate::fusion::on_write::{
    ir::{Arg, ElemwiseConfig, ElemwisePrecision, GlobalArgs, GlobalArgsLaunch},
    trace::{FuseOnWriteTrace, TraceRunner},
};

#[derive(new)]
/// Fuse element wise operations into a single kernel.
pub struct ElemwiseOptimization<R: JitRuntime> {
    trace: FuseOnWriteTrace,
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    len: usize,
}

#[derive(Serialize, Deserialize)]
/// State for the [elemwise optimization](ElemwiseOptimization).
pub struct ElemwiseOptimizationState {
    trace: FuseOnWriteTrace,
    len: usize,
}

impl<R: JitRuntime> ElemwiseOptimization<R> {
    /// Execute the optimization.
    pub fn execute<BT: BoolElement>(&mut self, context: &mut Context<'_, JitFusionHandle<R>>) {
        self.trace
            .run::<R, BT, ElemwiseRunner>(&self.client, &self.device, context, &ElemwiseRunner)
            .unwrap();
    }

    /// Number of element wise operations fused.
    pub fn num_ops_fused(&self) -> usize {
        self.len
    }

    /// Create an optimization from its [state](ElemwiseOptimizationState).
    pub fn from_state(device: &R::Device, state: ElemwiseOptimizationState) -> Self {
        Self {
            trace: state.trace,
            len: state.len,
            client: R::client(device),
            device: device.clone(),
        }
    }

    /// Convert the optimization to its [state](ElemwiseOptimizationState).
    pub fn to_state(&self) -> ElemwiseOptimizationState {
        ElemwiseOptimizationState {
            trace: self.trace.clone(),
            len: self.len,
        }
    }
}

pub struct ElemwiseRunner;

impl<R: JitRuntime> TraceRunner<R> for ElemwiseRunner {
    type Error = (); // No error possible

    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R::Server, R::Channel>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        config: &'a ElemwiseConfig,
    ) -> Result<(), Self::Error> {
        let arg = match config.ref_layout {
            Arg::Input(index, precision, _) => match precision {
                ElemwisePrecision::F32 => inputs.t_f32.values.get(index as usize),
                ElemwisePrecision::F16 => inputs.t_f16.values.get(index as usize),
                ElemwisePrecision::BF16 => inputs.t_bf16.values.get(index as usize),
                ElemwisePrecision::U64 => inputs.t_u64.values.get(index as usize),
                ElemwisePrecision::U32 => inputs.t_u32.values.get(index as usize),
                ElemwisePrecision::U16 => inputs.t_u16.values.get(index as usize),
                ElemwisePrecision::U8 => inputs.t_u8.values.get(index as usize),
                ElemwisePrecision::I64 => inputs.t_i64.values.get(index as usize),
                ElemwisePrecision::I32 => inputs.t_i32.values.get(index as usize),
                ElemwisePrecision::I16 => inputs.t_i16.values.get(index as usize),
                ElemwisePrecision::I8 => inputs.t_i8.values.get(index as usize),
                _ => panic!("Invalid value"),
            },
            Arg::Output(index, precision, _) => match precision {
                ElemwisePrecision::F32 => outputs.t_f32.values.get(index as usize),
                ElemwisePrecision::F16 => outputs.t_f16.values.get(index as usize),
                ElemwisePrecision::BF16 => outputs.t_bf16.values.get(index as usize),
                ElemwisePrecision::U64 => outputs.t_u64.values.get(index as usize),
                ElemwisePrecision::U32 => outputs.t_u32.values.get(index as usize),
                ElemwisePrecision::U16 => outputs.t_u16.values.get(index as usize),
                ElemwisePrecision::U8 => outputs.t_u8.values.get(index as usize),
                ElemwisePrecision::I64 => outputs.t_i64.values.get(index as usize),
                ElemwisePrecision::I32 => outputs.t_i32.values.get(index as usize),
                ElemwisePrecision::I16 => outputs.t_i16.values.get(index as usize),
                ElemwisePrecision::I8 => outputs.t_i8.values.get(index as usize),
                _ => panic!("Invalid value"),
            },
            _ => panic!("Invalid value"),
        };
        let (shape, vectorization) = match arg {
            Some(val) => match val {
                TensorArg::Handle {
                    handle,
                    vectorization_factor,
                } => (handle.shape, vectorization_factor),
                _ => panic!("Can't be an alias"),
            },
            None => panic!("Invalid argument"),
        };
        let total_elem = shape.iter().product::<usize>() / *vectorization as usize;
        let cube_dim = CubeDim::default();
        let cube_count = calculate_cube_count_elemwise(total_elem, cube_dim);

        unsafe {
            elemwise_fuse::launch_unchecked(
                client,
                cube_count,
                cube_dim,
                inputs,
                outputs,
                config.clone(),
            );
        };

        Ok(())
    }
}

#[cube(launch_unchecked)]
fn elemwise_fuse(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    #[comptime] config: &ElemwiseConfig,
) {
    // We write no values for this fusion.
    let values = Registry::<Arg, Line<f32>>::new();
    let args = comptime![Sequence::<Arg>::new()];
    let pos = ABSOLUTE_POS;

    let length = match comptime![config.ref_layout.clone()] {
        Arg::Input(index, precision, _) => match comptime![precision] {
            ElemwisePrecision::F32 => inputs.t_f32.index(index).len(),
            ElemwisePrecision::F16 => inputs.t_f16.index(index).len(),
            ElemwisePrecision::BF16 => inputs.t_bf16.index(index).len(),
            ElemwisePrecision::U64 => inputs.t_u64.index(index).len(),
            ElemwisePrecision::U32 => inputs.t_u32.index(index).len(),
            ElemwisePrecision::U16 => inputs.t_u16.index(index).len(),
            ElemwisePrecision::U8 => inputs.t_u8.index(index).len(),
            ElemwisePrecision::I64 => inputs.t_i64.index(index).len(),
            ElemwisePrecision::I32 => inputs.t_i32.index(index).len(),
            ElemwisePrecision::I16 => inputs.t_i16.index(index).len(),
            ElemwisePrecision::I8 => inputs.t_i8.index(index).len(),
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        Arg::Output(index, precision, _) => match comptime![precision] {
            ElemwisePrecision::F32 => outputs.t_f32.index(index).len(),
            ElemwisePrecision::F16 => outputs.t_f16.index(index).len(),
            ElemwisePrecision::BF16 => outputs.t_bf16.index(index).len(),
            ElemwisePrecision::U64 => outputs.t_u64.index(index).len(),
            ElemwisePrecision::U32 => outputs.t_u32.index(index).len(),
            ElemwisePrecision::U16 => outputs.t_u16.index(index).len(),
            ElemwisePrecision::U8 => outputs.t_u8.index(index).len(),
            ElemwisePrecision::I64 => outputs.t_i64.index(index).len(),
            ElemwisePrecision::I32 => outputs.t_i32.index(index).len(),
            ElemwisePrecision::I16 => outputs.t_i16.index(index).len(),
            ElemwisePrecision::I8 => outputs.t_i8.index(index).len(),
            _ => comptime![panic!("Unsupported precision {precision:?}")],
        },
        _ => comptime![panic!("Invalid ref layout.")],
    };

    if pos < length {
        fuse_on_write::<f32>(inputs, outputs, pos, values, args, config)
    }
}
