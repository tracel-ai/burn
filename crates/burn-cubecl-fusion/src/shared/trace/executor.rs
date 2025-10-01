use std::marker::PhantomData;

use burn_fusion::stream::{Context, ScalarId};
use burn_ir::ScalarIr;
use burn_tensor::DType;
use cubecl::{
    CubeElement, Runtime,
    client::ComputeClient,
    prelude::{ScalarArg, Sequence, TensorArg},
};

use super::{
    FuseResources, HandleOutput, LaunchPlan, ReferenceSelection, TensorView, TraceError,
    TraceRunner, TuneOutput, block::FuseBlock,
};
use crate::{
    CubeFusionHandle, elem_dtype,
    shared::{
        ir::{FuseBlockConfig, FuseOp, FusePrecision, GlobalArgsLaunch, RefLayout, VirtualLayout},
        tensor::{GlobalScalar, GlobalTensorArg},
        trace::HandleInput,
    },
};

/// Execute a [plan](LaunchPlan) using a [runner](TraceRunner) modifying the [context](Context).
pub struct LaunchPlanExecutor<'a, R: Runtime> {
    resources: &'a FuseResources,
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
    pub fn new(resources: &'a FuseResources, blocks: &'a Vec<FuseBlock>) -> Self {
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
    ) -> Result<TuneOutput<R>, ExecutionError<R, Runner>> {
        let mut num_writes = 0;
        for b in plan.blocks.iter() {
            num_writes += b.writes.len();
        }

        #[cfg(feature = "autotune-checks")]
        let mut tune_output = TuneOutput::Checked {
            handles: std::collections::HashMap::new(),
        };

        #[cfg(not(feature = "autotune-checks"))]
        let mut tune_output = TuneOutput::UnChecked(PhantomData);

        if num_writes == 0 {
            // Nothing to write, can skip execution.
            return Ok(tune_output);
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
        register_outputs::<BT, R>(&plan.handle_outputs, &mut outputs, &mut tune_output);

        let mut configs = Vec::with_capacity(plan.blocks.len());

        for (block_plan, block) in plan.blocks.into_iter().zip(self.blocks) {
            let reference = match block_plan.reference {
                ReferenceSelection::Concrete { layout, .. } => RefLayout::Concrete(layout),
                ReferenceSelection::VirtualShape { original, .. } => {
                    RefLayout::Virtual(VirtualLayout::Shape(original, block_plan.width as u32))
                }
                ReferenceSelection::SwapDims { original, dims } => {
                    RefLayout::Virtual(VirtualLayout::SwapDims(original, dims))
                }
                ReferenceSelection::Reshaped { reshape_pos } => {
                    RefLayout::Virtual(VirtualLayout::Reshaped {
                        reshape_pos: reshape_pos as u32,
                        line_size: block_plan.width as u32,
                    })
                }
                ReferenceSelection::NotFound | ReferenceSelection::Searching => {
                    return Err(ExecutionError::new(
                        TraceError::ReferenceNotFound,
                        plan.handle_inputs,
                        plan.handle_outputs,
                    ));
                }
            };

            let mut ops = Sequence::<FuseOp>::new();

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

            let config = FuseBlockConfig {
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
        })?;

        Ok(tune_output)
    }
}

fn register_inputs<'h, R: Runtime>(
    handle_inputs: &'h [HandleInput<R>],
    inputs: &mut GlobalArgsLaunch<'h, R>,
) {
    for hi in handle_inputs.iter() {
        match hi {
            HandleInput::Normal(hi) => {
                let arg = hi
                    .handle
                    .as_tensor_arg(&hi.global_ir.shape, hi.vectorization);
                inputs.tensors.push(GlobalTensorArg::new(
                    arg,
                    hi.precision.into_elem(),
                    hi.broadcated,
                ));
            }
            HandleInput::QuantValues(hi) => {
                let arg = hi
                    .handle
                    .as_tensor_arg(&hi.global_ir.shape, hi.vectorization);
                inputs
                    .tensors
                    .push(GlobalTensorArg::new(arg, hi.precision.into_elem(), false));
            }
            HandleInput::QuantParams(hi) => {
                let arg = hi.handle.as_tensor_arg(&hi.shape, 1);
                inputs
                    .tensors
                    .push(GlobalTensorArg::new(arg, hi.precision.into_elem(), false));
            }
        }
    }
}

fn register_outputs<'s, BT: CubeElement, R: Runtime>(
    handle_outputs: &'s [HandleOutput<R>],
    outputs: &mut GlobalArgsLaunch<'s, R>,
    #[allow(unused_variables)] tune_output: &mut TuneOutput<R>,
) {
    for item in handle_outputs.iter() {
        match item {
            HandleOutput::Alias {
                input_pos,
                precision,
                #[cfg(feature = "autotune-checks")]
                debug_info,
            } => {
                outputs.tensors.push(GlobalTensorArg::new(
                    TensorArg::alias(*input_pos),
                    precision.into_elem(),
                    false,
                ));

                #[cfg(feature = "autotune-checks")]
                if let TuneOutput::Checked { handles, .. } = tune_output {
                    handles.insert(
                        debug_info.relative_id,
                        (debug_info.global_shape.clone(), debug_info.handle.clone()),
                    );
                }
            }
            HandleOutput::Owned {
                precision,
                handle,
                global_shape,
                vectorization,
                #[cfg(feature = "autotune-checks")]
                relative_id,
                ..
            } => {
                let arg = handle.as_tensor_arg(global_shape, *vectorization);

                let elem = match precision {
                    FusePrecision::Bool => match elem_dtype::<BT>() {
                        DType::U32 => FusePrecision::U32.into_elem(),
                        DType::U8 => FusePrecision::U8.into_elem(),
                        _ => todo!(),
                    },
                    _ => precision.into_elem(),
                };

                #[cfg(feature = "autotune-checks")]
                if let TuneOutput::Checked { handles, .. } = tune_output {
                    handles.insert(*relative_id, (global_shape.clone(), handle.clone()));
                }

                outputs.tensors.push(GlobalTensorArg::new(arg, elem, false));
            }
        }
    }
}

fn register_scalars<'h, R: Runtime>(
    scalars: impl Iterator<Item = &'h (FusePrecision, u64)>,
    views: impl DoubleEndedIterator<Item = &'h TensorView>,
    context: &mut Context<'_, CubeFusionHandle<R>>,
    inputs: &mut GlobalArgsLaunch<'h, R>,
) {
    for (precision, id) in scalars {
        match precision {
            FusePrecision::F64 => {
                inputs.scalars.push(GlobalScalar::F64(
                    match context.scalars.get(&ScalarId { value: *id }) {
                        Some(ScalarIr::F64(val)) => *val,
                        _ => panic!(),
                    },
                ));
            }
            FusePrecision::F32 | FusePrecision::Flex32 => {
                inputs.scalars.push(GlobalScalar::F32(
                    match context.scalars.get(&ScalarId { value: *id }) {
                        Some(ScalarIr::F32(val)) => *val,
                        _ => panic!(),
                    },
                ));
            }
            FusePrecision::F16 => {
                inputs.scalars.push(GlobalScalar::F16(
                    match context.scalars.get(&ScalarId { value: *id }) {
                        Some(ScalarIr::F16(val)) => *val,
                        _ => panic!(),
                    },
                ));
            }
            FusePrecision::BF16 => {
                inputs.scalars.push(GlobalScalar::BF16(
                    match context.scalars.get(&ScalarId { value: *id }) {
                        Some(ScalarIr::BF16(val)) => *val,
                        _ => panic!(),
                    },
                ));
            }
            FusePrecision::I64 => {
                inputs.scalars.push(GlobalScalar::I64(
                    match context.scalars.get(&ScalarId { value: *id }) {
                        Some(ScalarIr::I64(val)) => *val,
                        _ => panic!(),
                    },
                ));
            }
            FusePrecision::I32 => {
                inputs.scalars.push(GlobalScalar::I32(
                    match context.scalars.get(&ScalarId { value: *id }) {
                        Some(ScalarIr::I32(val)) => *val,
                        _ => panic!(),
                    },
                ));
            }
            FusePrecision::I16 => {
                inputs.scalars.push(GlobalScalar::I16(
                    match context.scalars.get(&ScalarId { value: *id }) {
                        Some(ScalarIr::I16(val)) => *val,
                        _ => panic!(),
                    },
                ));
            }
            FusePrecision::I8 => {
                inputs.scalars.push(GlobalScalar::I8(
                    match context.scalars.get(&ScalarId { value: *id }) {
                        Some(ScalarIr::I8(val)) => *val,
                        _ => panic!(),
                    },
                ));
            }
            FusePrecision::U64 => {
                inputs.scalars.push(GlobalScalar::U64(
                    match context.scalars.get(&ScalarId { value: *id }) {
                        Some(ScalarIr::U64(val)) => *val,
                        _ => panic!(),
                    },
                ));
            }
            FusePrecision::U32 => {
                inputs.scalars.push(GlobalScalar::U32(
                    match context.scalars.get(&ScalarId { value: *id }) {
                        Some(ScalarIr::U32(val)) => *val,
                        _ => panic!(),
                    },
                ));
            }
            FusePrecision::U16 => {
                inputs.scalars.push(GlobalScalar::U16(
                    match context.scalars.get(&ScalarId { value: *id }) {
                        Some(ScalarIr::U16(val)) => *val,
                        _ => panic!(),
                    },
                ));
            }
            FusePrecision::U8 => {
                inputs.scalars.push(GlobalScalar::U8(
                    match context.scalars.get(&ScalarId { value: *id }) {
                        Some(ScalarIr::U8(val)) => *val,
                        _ => panic!(),
                    },
                ));
            }
            FusePrecision::Bool => todo!(),
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
