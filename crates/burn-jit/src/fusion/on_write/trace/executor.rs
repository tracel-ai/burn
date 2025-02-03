use std::{collections::BTreeMap, marker::PhantomData};

use burn_fusion::stream::Context;
use burn_tensor::DType;
use cubecl::{
    client::ComputeClient,
    prelude::{ScalarArg, Sequence, TensorArg},
};

use super::{HandleInput, HandleOutput, LaunchPlan, Reshape, TraceRunner};
use crate::{
    fusion::{
        on_write::ir::{ElemwiseConfig, ElemwiseOp, ElemwisePrecision, GlobalArgsLaunch},
        JitFusionHandle,
    },
    BoolElement, JitRuntime,
};

/// Execute a [plan](LaunchPlan) using a [runner](TraceRunner) modifying the [context](Context).
pub struct LaunchPlanExecutor<'a, R: JitRuntime> {
    scalars: &'a BTreeMap<ElemwisePrecision, u32>,
    reshapes: &'a Vec<Reshape>,
    ops: &'a Vec<ElemwiseOp>,
    _r: PhantomData<R>,
}

#[derive(new)]
pub struct ExecutionError<R: JitRuntime, Runner: TraceRunner<R>> {
    pub runner_error: Runner::Error,
    pub handles_input: Vec<HandleInput<R>>,
    pub handles_output: Vec<HandleOutput<R>>,
}

impl<'a, R: JitRuntime> LaunchPlanExecutor<'a, R> {
    pub fn new(
        scalars: &'a BTreeMap<ElemwisePrecision, u32>,
        reshapes: &'a Vec<Reshape>,
        ops: &'a Vec<ElemwiseOp>,
    ) -> Self {
        Self {
            scalars,
            reshapes,
            ops,
            _r: PhantomData,
        }
    }

    pub fn execute<Runner: TraceRunner<R>, BT: BoolElement>(
        self,
        client: &ComputeClient<R::Server, R::Channel>,
        runner: &Runner,
        context: &mut Context<'_, JitFusionHandle<R>>,
        plan: LaunchPlan<'a, R>,
    ) -> Result<(), ExecutionError<R, Runner>> {
        let reference = match plan.reference {
            Some(reference) => reference,
            None => {
                if plan.writes.is_empty() {
                    // Nothing to write, can skip execution.
                    return Ok(());
                } else {
                    panic!("An output should exist for the fused kernel")
                }
            }
        };

        let inputs = self.register_inputs(context, &plan.handle_inputs);
        let outputs = self.register_outputs::<BT>(&plan.handle_outputs);

        let mut ops = Sequence::<ElemwiseOp>::new();

        for read_ops in plan.reads.into_values() {
            for op in read_ops {
                ops.push(op);
            }
        }

        for op in self.ops.iter() {
            ops.push(op.clone());
        }

        for op in plan.writes.into_values() {
            ops.push(op);
        }

        let config = ElemwiseConfig {
            rank: plan.rank as u32,
            ref_layout: reference.layout,
            ops,
        };

        Runner::run(runner, client, inputs, outputs, &config)
            .map_err(|err| ExecutionError::new(err, plan.handle_inputs, plan.handle_outputs))
    }

    fn register_inputs<'h>(
        &self,
        context: &mut Context<'_, JitFusionHandle<R>>,
        handle_inputs: &'h [HandleInput<R>],
    ) -> GlobalArgsLaunch<'h, R> {
        let mut inputs = GlobalArgsLaunch::default();

        for hi in handle_inputs.iter() {
            let arg = hi.handle.as_tensor_arg(&hi.global_shape, hi.vectorization);
            match hi.precision {
                ElemwisePrecision::F32 => inputs.t_f32.push(arg),
                ElemwisePrecision::F16 => inputs.t_f16.push(arg),
                ElemwisePrecision::BF16 => inputs.t_bf16.push(arg),
                ElemwisePrecision::I64 => inputs.t_i64.push(arg),
                ElemwisePrecision::I32 => inputs.t_i32.push(arg),
                ElemwisePrecision::I16 => inputs.t_i16.push(arg),
                ElemwisePrecision::I8 => inputs.t_i8.push(arg),
                ElemwisePrecision::U64 => inputs.t_u64.push(arg),
                ElemwisePrecision::U32 => inputs.t_u32.push(arg),
                ElemwisePrecision::U16 => inputs.t_u16.push(arg),
                ElemwisePrecision::U8 => inputs.t_u8.push(arg),
                _ => panic!("Unsupported input precision {:?}", hi.precision),
            };
        }

        for (precision, count) in self.scalars.iter() {
            for i in 0..(*count as usize) {
                match precision {
                    ElemwisePrecision::F32 => {
                        inputs.s_f32.push(ScalarArg::new(context.scalar_f32[i]))
                    }
                    ElemwisePrecision::F16 => {
                        inputs.s_f16.push(ScalarArg::new(context.scalar_f16[i]))
                    }
                    ElemwisePrecision::BF16 => {
                        inputs.s_bf16.push(ScalarArg::new(context.scalar_bf16[i]))
                    }
                    ElemwisePrecision::I64 => {
                        inputs.s_i64.push(ScalarArg::new(context.scalar_i64[i]))
                    }
                    ElemwisePrecision::I32 => {
                        inputs.s_i32.push(ScalarArg::new(context.scalar_i32[i]))
                    }
                    ElemwisePrecision::I16 => {
                        inputs.s_i16.push(ScalarArg::new(context.scalar_i16[i]))
                    }
                    ElemwisePrecision::I8 => inputs.s_i8.push(ScalarArg::new(context.scalar_i8[i])),
                    ElemwisePrecision::U64 => {
                        inputs.s_u64.push(ScalarArg::new(context.scalar_u64[i]))
                    }
                    ElemwisePrecision::U32 => {
                        inputs.s_u32.push(ScalarArg::new(context.scalar_u32[i]))
                    }
                    ElemwisePrecision::U16 => {
                        inputs.s_u16.push(ScalarArg::new(context.scalar_u16[i]))
                    }
                    ElemwisePrecision::U8 => inputs.s_u8.push(ScalarArg::new(context.scalar_u8[i])),
                    ElemwisePrecision::Bool => todo!(),
                }
            }
        }

        // Reshape values are pushed in reverse in the same scalar buffer for all `u32`
        for relative in self.reshapes.iter().rev() {
            let global = context.tensors.get(&relative.reshaped).unwrap();

            for shape in global.shape.iter().rev() {
                inputs.s_u32.push(ScalarArg::new(*shape as u32))
            }
        }

        inputs
    }

    fn register_outputs<'s, BT: BoolElement>(
        &self,
        handle_outputs: &'s [HandleOutput<R>],
    ) -> GlobalArgsLaunch<'s, R> {
        let mut outputs = GlobalArgsLaunch::default();

        for item in handle_outputs.iter() {
            match item {
                HandleOutput::Alias {
                    input_pos,
                    precision,
                } => match precision {
                    ElemwisePrecision::F32 => outputs.t_f32.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::F16 => outputs.t_f16.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::BF16 => outputs.t_bf16.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::I64 => outputs.t_i64.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::I32 => outputs.t_i32.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::I16 => outputs.t_i16.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::I8 => outputs.t_i8.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::U64 => outputs.t_u64.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::U32 => outputs.t_u32.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::U16 => outputs.t_u16.push(TensorArg::alias(*input_pos)),
                    ElemwisePrecision::U8 => outputs.t_u8.push(TensorArg::alias(*input_pos)),
                    _ => todo!(),
                },
                HandleOutput::Owned {
                    precision,
                    handle,
                    global_shape,
                    vectorization,
                    ..
                } => {
                    let arg = handle.as_tensor_arg(global_shape, *vectorization);

                    match precision {
                        ElemwisePrecision::F32 => outputs.t_f32.push(arg),
                        ElemwisePrecision::F16 => outputs.t_f16.push(arg),
                        ElemwisePrecision::BF16 => outputs.t_bf16.push(arg),
                        ElemwisePrecision::I64 => outputs.t_i64.push(arg),
                        ElemwisePrecision::I32 => outputs.t_i32.push(arg),
                        ElemwisePrecision::I16 => outputs.t_i16.push(arg),
                        ElemwisePrecision::I8 => outputs.t_i8.push(arg),
                        ElemwisePrecision::U64 => outputs.t_u64.push(arg),
                        ElemwisePrecision::U32 => outputs.t_u32.push(arg),
                        ElemwisePrecision::U16 => outputs.t_u16.push(arg),
                        ElemwisePrecision::U8 => outputs.t_u8.push(arg),
                        ElemwisePrecision::Bool => match BT::dtype() {
                            DType::U32 => outputs.t_u32.push(arg),
                            DType::U8 => outputs.t_u8.push(arg),
                            _ => todo!(),
                        },
                    };
                }
            }
        }

        outputs
    }
}
