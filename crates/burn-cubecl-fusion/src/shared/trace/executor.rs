use std::marker::PhantomData;

use burn_fusion::stream::Context;
use burn_tensor::DType;
use cubecl::{
    client::ComputeClient,
    prelude::{ScalarArg, Sequence, TensorArg},
    CubeElement, Runtime,
};

use super::{
    HandleInput, HandleOutput, LaunchPlan, MultiTraceRunner, ReferenceSelection, TensorView,
    TraceError, TraceRunner,
};
use crate::{
    elem_dtype,
    shared::{
        ir::{ElemwiseConfig, ElemwiseOp, ElemwisePrecision, GlobalArgsLaunch, RefLayout},
        tensor::{GlobalScalar, GlobalTensorArg},
    },
    CubeFusionHandle,
};

/// Execute a [plan](LaunchPlan) using a [runner](TraceRunner) modifying the [context](Context).
pub struct LaunchPlanExecutor<'a, R: Runtime> {
    scalars: &'a Vec<(ElemwisePrecision, u32)>,
    views: &'a Vec<TensorView>,
    ops: &'a Vec<ElemwiseOp>,
    _r: PhantomData<R>,
}

/// Execute a [plan](LaunchPlan) using a [runner](TraceRunner) modifying the [context](Context).
#[allow(clippy::type_complexity)]
pub struct LaunchMultiPlanExecutor<'a, R: Runtime> {
    scalars: (
        &'a Vec<(ElemwisePrecision, u32)>,
        &'a Vec<(ElemwisePrecision, u32)>,
    ),
    views: (&'a Vec<TensorView>, &'a Vec<TensorView>),
    ops: (&'a Vec<ElemwiseOp>, &'a Vec<ElemwiseOp>),
    _r: PhantomData<R>,
}

#[derive(new, Debug)]
pub struct ExecutionError<R: Runtime, Runner: TraceRunner<R>> {
    pub error: TraceError<Runner::Error>,
    pub handles_input: Vec<HandleInput<R>>,
    pub handles_output: Vec<HandleOutput<R>>,
}

#[derive(new, Debug)]
pub struct MultiExecutionError<R: Runtime, Runner: MultiTraceRunner<R>> {
    pub error: TraceError<Runner::Error>,
    pub plan_0_handles_input: Vec<HandleInput<R>>,
    pub plan_0_handles_output: Vec<HandleOutput<R>>,
    pub plan_1_handles_input: Vec<HandleInput<R>>,
    pub plan_1_handles_output: Vec<HandleOutput<R>>,
}

impl<'a, R: Runtime> LaunchMultiPlanExecutor<'a, R> {
    #[allow(clippy::type_complexity)]
    pub fn new(
        scalars: (
            &'a Vec<(ElemwisePrecision, u32)>,
            &'a Vec<(ElemwisePrecision, u32)>,
        ),
        views: (&'a Vec<TensorView>, &'a Vec<TensorView>),
        ops: (&'a Vec<ElemwiseOp>, &'a Vec<ElemwiseOp>),
    ) -> Self {
        Self {
            scalars,
            views,
            ops,
            _r: PhantomData,
        }
    }

    pub fn execute<Runner: MultiTraceRunner<R>, BT: CubeElement>(
        self,
        client: &ComputeClient<R::Server, R::Channel>,
        runner: &Runner,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        mut plans: (LaunchPlan<'a, R>, LaunchPlan<'a, R>),
    ) -> Result<(), MultiExecutionError<R, Runner>> {
        if plans.0.writes.is_empty() && plans.1.writes.is_empty() {
            // Nothing to write, can skip execution.
            return Ok(());
        }
        let reference = match plans.0.reference {
            ReferenceSelection::Found(reference) => RefLayout::Concrete(reference.layout),
            ReferenceSelection::Virtual(shape) => RefLayout::Virtual(shape),
            ReferenceSelection::Searching | ReferenceSelection::NotFound => {
                return Err(MultiExecutionError::new(
                    TraceError::ReferenceNotFound,
                    plans.0.handle_inputs,
                    plans.0.handle_outputs,
                    plans.1.handle_inputs,
                    plans.1.handle_outputs,
                ))
            }
        };

        let mut inputs = GlobalArgsLaunch::default();
        let mut outputs = GlobalArgsLaunch::default();

        register_inputs(&plans.0.handle_inputs, &mut inputs);
        register_outputs::<BT, R>(&plans.0.handle_outputs, &mut outputs);

        let output_offset = outputs.tensors.values.len() as u32;

        let mut ops = Sequence::<ElemwiseOp>::new();

        for read_ops in plans.0.reads.into_values() {
            for op in read_ops {
                ops.push(op);
            }
        }

        for op in self.ops.0.iter() {
            ops.push(op.clone());
        }

        for op in plans.0.writes.into_values() {
            ops.push(op);
        }

        let config_0 = ElemwiseConfig {
            rank: plans.0.rank as u32,
            ref_layout: reference,
            ops,
            width: plans.0.width,
        };

        plans.1.output_offset(output_offset);

        let reference = match plans.1.reference {
            ReferenceSelection::Found(reference) => RefLayout::Concrete(reference.layout),
            ReferenceSelection::Searching | ReferenceSelection::NotFound => {
                return Err(MultiExecutionError::new(
                    TraceError::ReferenceNotFound,
                    plans.0.handle_inputs,
                    plans.0.handle_outputs,
                    plans.1.handle_inputs,
                    plans.1.handle_outputs,
                ))
            }
            ReferenceSelection::Virtual(shape) => RefLayout::Virtual(shape),
        };

        register_inputs(&plans.1.handle_inputs, &mut inputs);
        register_outputs::<BT, R>(&plans.1.handle_outputs, &mut outputs);
        register_scalars::<R>(
            self.scalars.0.iter().chain(self.scalars.1.iter()),
            self.views.0.iter().chain(self.views.1.iter()),
            context,
            &mut inputs,
        );

        let mut ops = Sequence::<ElemwiseOp>::new();

        for read_ops in plans.1.reads.into_values() {
            for op in read_ops {
                ops.push(op);
            }
        }

        for op in self.ops.1.iter() {
            ops.push(op.clone());
        }

        for op in plans.1.writes.into_values() {
            ops.push(op);
        }
        let config_1 = ElemwiseConfig {
            rank: plans.1.rank as u32,
            ref_layout: reference,
            ops,
            width: plans.1.width,
        };

        Runner::run(runner, client, inputs, outputs, &config_0, &config_1).map_err(|err| {
            MultiExecutionError::new(
                TraceError::RunnerError(err),
                plans.0.handle_inputs,
                plans.0.handle_outputs,
                plans.1.handle_inputs,
                plans.1.handle_outputs,
            )
        })
    }
}
impl<'a, R: Runtime> LaunchPlanExecutor<'a, R> {
    pub fn new(
        scalars: &'a Vec<(ElemwisePrecision, u32)>,
        views: &'a Vec<TensorView>,
        ops: &'a Vec<ElemwiseOp>,
    ) -> Self {
        Self {
            scalars,
            views,
            ops,
            _r: PhantomData,
        }
    }

    pub fn execute<Runner: TraceRunner<R>, BT: CubeElement>(
        self,
        client: &ComputeClient<R::Server, R::Channel>,
        runner: &Runner,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        plan: LaunchPlan<'a, R>,
    ) -> Result<(), ExecutionError<R, Runner>> {
        if plan.writes.is_empty() {
            // Nothing to write, can skip execution.
            return Ok(());
        }

        let reference = match plan.reference {
            ReferenceSelection::Found(reference) => RefLayout::Concrete(reference.layout),
            ReferenceSelection::Virtual(shape) => RefLayout::Virtual(shape),
            ReferenceSelection::NotFound | ReferenceSelection::Searching => {
                return Err(ExecutionError::new(
                    TraceError::ReferenceNotFound,
                    plan.handle_inputs,
                    plan.handle_outputs,
                ));
            }
        };

        let mut inputs = GlobalArgsLaunch::default();
        let mut outputs = GlobalArgsLaunch::default();

        register_inputs(&plan.handle_inputs, &mut inputs);
        register_scalars(self.scalars.iter(), self.views.iter(), context, &mut inputs);
        register_outputs::<BT, R>(&plan.handle_outputs, &mut outputs);

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
            ref_layout: reference,
            ops,
            width: plan.width,
        };

        Runner::run(runner, client, inputs, outputs, &config).map_err(|err| {
            ExecutionError::new(
                TraceError::RunnerError(err),
                plan.handle_inputs,
                plan.handle_outputs,
            )
        })
    }
}

fn register_inputs<'h, R: Runtime>(
    handle_inputs: &'h [HandleInput<R>],
    inputs: &mut GlobalArgsLaunch<'h, R>,
) {
    for hi in handle_inputs.iter() {
        let arg = hi.handle.as_tensor_arg(&hi.global_shape, hi.vectorization);
        inputs.tensors.push(GlobalTensorArg::new(
            arg,
            hi.precision.into_elem(),
            hi.broadcated,
        ));
    }
}

fn register_outputs<'s, BT: CubeElement, R: Runtime>(
    handle_outputs: &'s [HandleOutput<R>],
    outputs: &mut GlobalArgsLaunch<'s, R>,
) {
    for item in handle_outputs.iter() {
        match item {
            HandleOutput::Alias {
                input_pos,
                precision,
            } => {
                outputs.tensors.push(GlobalTensorArg::new(
                    TensorArg::alias(*input_pos),
                    precision.into_elem(),
                    false,
                ));
            }
            HandleOutput::Owned {
                precision,
                handle,
                global_shape,
                vectorization,
                ..
            } => {
                let arg = handle.as_tensor_arg(global_shape, *vectorization);

                let elem = match precision {
                    ElemwisePrecision::Bool => match elem_dtype::<BT>() {
                        DType::U32 => ElemwisePrecision::U32.into_elem(),
                        DType::U8 => ElemwisePrecision::U8.into_elem(),
                        _ => todo!(),
                    },
                    _ => precision.into_elem(),
                };
                outputs.tensors.push(GlobalTensorArg::new(arg, elem, false));
            }
        }
    }
}

fn register_scalars<'h, R: Runtime>(
    scalars: impl Iterator<Item = &'h (ElemwisePrecision, u32)>,
    views: impl DoubleEndedIterator<Item = &'h TensorView>,
    context: &mut Context<'_, CubeFusionHandle<R>>,
    inputs: &mut GlobalArgsLaunch<'h, R>,
) {
    let mut index_f32 = 0;
    let mut index_f16 = 0;
    let mut index_bf16 = 0;
    let mut index_u64 = 0;
    let mut index_u32 = 0;
    let mut index_u16 = 0;
    let mut index_u8 = 0;
    let mut index_i64 = 0;
    let mut index_i32 = 0;
    let mut index_i16 = 0;
    let mut index_i8 = 0;

    for (precision, _pos) in scalars {
        match precision {
            ElemwisePrecision::F32 => {
                inputs
                    .scalars
                    .push(GlobalScalar::F32(context.scalar_f32[index_f32]));
                index_f32 += 1;
            }
            ElemwisePrecision::F16 => {
                inputs
                    .scalars
                    .push(GlobalScalar::F16(context.scalar_f16[index_f16]));
                index_f16 += 1;
            }
            ElemwisePrecision::BF16 => {
                inputs
                    .scalars
                    .push(GlobalScalar::BF16(context.scalar_bf16[index_bf16]));
                index_bf16 += 1;
            }
            ElemwisePrecision::I64 => {
                inputs
                    .scalars
                    .push(GlobalScalar::I64(context.scalar_i64[index_i64]));
                index_i64 += 1;
            }
            ElemwisePrecision::I32 => {
                inputs
                    .scalars
                    .push(GlobalScalar::I32(context.scalar_i32[index_i32]));
                index_i32 += 1;
            }
            ElemwisePrecision::I16 => {
                inputs
                    .scalars
                    .push(GlobalScalar::I16(context.scalar_i16[index_i16]));
                index_i16 += 1;
            }
            ElemwisePrecision::I8 => {
                inputs
                    .scalars
                    .push(GlobalScalar::I8(context.scalar_i8[index_i8]));
                index_i8 += 1;
            }
            ElemwisePrecision::U64 => {
                inputs
                    .scalars
                    .push(GlobalScalar::U64(context.scalar_u64[index_u64]));
                index_u64 += 1;
            }
            ElemwisePrecision::U32 => {
                inputs
                    .scalars
                    .push(GlobalScalar::U32(context.scalar_u32[index_u32]));
                index_u32 += 1;
            }
            ElemwisePrecision::U16 => {
                inputs
                    .scalars
                    .push(GlobalScalar::U16(context.scalar_u16[index_u16]));
                index_u16 += 1;
            }
            ElemwisePrecision::U8 => {
                inputs
                    .scalars
                    .push(GlobalScalar::U8(context.scalar_u8[index_u8]));
                index_u8 += 1;
            }
            ElemwisePrecision::Bool => todo!(),
        }
    }

    for relative in views {
        if let TensorView::Reshape { reshaped, .. } = relative {
            let global = context.tensors.get(reshaped).unwrap();

            for shape in global.shape.iter() {
                inputs.reshapes.push(ScalarArg::new(*shape as u32));
            }
        }
    }
}
