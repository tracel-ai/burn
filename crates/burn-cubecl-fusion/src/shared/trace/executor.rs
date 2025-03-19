use std::marker::PhantomData;

use burn_fusion::stream::Context;
use burn_tensor::DType;
use cubecl::{
    client::ComputeClient,
    prelude::{ScalarArg, Sequence, TensorArg},
    CubeElement, Runtime,
};

use super::{
    block::FuseBlock, HandleInput, HandleOutput, KernelResources, LaunchPlan, ReferenceSelection,
    TensorView, TraceError, TraceRunner,
};
use crate::{
    elem_dtype,
    shared::{
        ir::{
            ElemwiseConfig, ElemwiseOp, ElemwisePrecision, GlobalArgsLaunch, RefLayout,
            VirtualLayout,
        },
        tensor::{GlobalScalar, GlobalTensorArg},
    },
    CubeFusionHandle,
};

/// Execute a [plan](LaunchPlan) using a [runner](TraceRunner) modifying the [context](Context).
pub struct LaunchPlanExecutor<'a, R: Runtime> {
    resources: &'a KernelResources,
    blocks: &'a Vec<FuseBlock>,
    _r: PhantomData<R>,
}

#[derive(new, Debug)]
pub struct ExecutionError<R: Runtime, Runner: TraceRunner<R>> {
    pub error: TraceError<Runner::Error>,
    pub handles_input: Vec<HandleInput<R>>,
    pub handles_output: Vec<HandleOutput<R>>,
}

impl<'a, R: Runtime> LaunchPlanExecutor<'a, R> {
    pub fn new(resources: &'a KernelResources, blocks: &'a Vec<FuseBlock>) -> Self {
        Self {
            resources,
            blocks,
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
        println!("Plan {plan:?}");
        let mut num_writes = 0;
        for b in plan.blocks.iter() {
            num_writes += b.writes.len();
        }

        if num_writes == 0 {
            // Nothing to write, can skip execution.
            println!("Nothing to execute");
            return Ok(());
        }

        let mut inputs = GlobalArgsLaunch::default();
        let mut outputs = GlobalArgsLaunch::default();

        register_inputs(&plan.handle_inputs, &mut inputs);
        register_scalars(
            self.resources.scalars.iter(),
            self.resources.views.iter(),
            context,
            &mut inputs,
        );
        register_outputs::<BT, R>(&plan.handle_outputs, &mut outputs);

        let mut configs = Vec::with_capacity(plan.blocks.len());

        for (block_plan, block) in plan.blocks.into_iter().zip(self.blocks) {
            let reference = match block_plan.reference {
                ReferenceSelection::Concrete { layout, .. } => RefLayout::Concrete(layout),
                ReferenceSelection::SwapDims { original, dims } => {
                    RefLayout::Virtual(VirtualLayout::SwapDims(original, dims))
                }
                ReferenceSelection::Reshaped { reshape_pos } => {
                    RefLayout::Virtual(VirtualLayout::Reshaped(reshape_pos as u32))
                }
                ReferenceSelection::NotFound | ReferenceSelection::Searching => {
                    return Err(ExecutionError::new(
                        TraceError::ReferenceNotFound,
                        plan.handle_inputs,
                        plan.handle_outputs,
                    ));
                }
            };

            let mut ops = Sequence::<ElemwiseOp>::new();

            for read_ops in block_plan.reads.into_values() {
                for op in read_ops {
                    ops.push(op);
                }
            }

            for op in block.ops.iter() {
                ops.push(op.clone());
            }

            for op in block_plan.writes.into_values() {
                ops.push(op);
            }

            let config = ElemwiseConfig {
                rank: plan.rank as u32,
                ref_layout: reference,
                ops,
                width: block_plan.width,
            };
            configs.push(config);
        }

        Runner::run(runner, client, inputs, outputs, &configs).map_err(|err| {
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
