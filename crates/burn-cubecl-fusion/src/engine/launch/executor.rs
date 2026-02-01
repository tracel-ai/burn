use super::{HandleInput, HandleOutput, LaunchPlan, ReferenceSelection};
use crate::engine::launch::runner::TraceRunner;
use crate::engine::trace::{FuseResources, TensorView, TraceError, TuneOutput, block::FuseBlock};
use crate::{
    CubeFusionHandle, elem_dtype,
    engine::{
        codegen::ir::{
            FuseBlockConfig, FuseOp, FuseType, GlobalArgsLaunch, RefLayout, VirtualLayout,
        },
        codegen::tensor::GlobalTensorArg,
    },
};
use burn_fusion::stream::{Context, ScalarId};
use burn_ir::ScalarIr;
use burn_std::DType;
use cubecl::{
    CubeElement, Runtime,
    client::ComputeClient,
    prelude::{InputScalar, ScalarArg, TensorArg},
};
use std::marker::PhantomData;

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
        client: &ComputeClient<R>,
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

        for layout in plan.runtime_layouts {
            for s in layout.shape {
                inputs.runtime_layouts.push(ScalarArg::new(s));
            }
            for s in layout.strides {
                inputs.runtime_layouts.push(ScalarArg::new(s));
            }
        }

        let mut configs = Vec::with_capacity(plan.blocks.len());

        for (block_plan, block) in plan.blocks.into_iter().zip(self.blocks) {
            let reference = match block_plan.reference {
                ReferenceSelection::Concrete { layout, .. } => RefLayout::Concrete(layout),
                ReferenceSelection::VirtualShape { original, .. } => {
                    RefLayout::Virtual(VirtualLayout::Shape(original, block_plan.width))
                }
                ReferenceSelection::SwapDims { original, dims } => {
                    RefLayout::Virtual(VirtualLayout::SwapDims(original, dims))
                }
                ReferenceSelection::Reshaped { reshape_pos } => {
                    RefLayout::Virtual(VirtualLayout::Reshaped {
                        reshape_pos,
                        line_size: block_plan.width,
                    })
                }
                ReferenceSelection::Runtime { pos } => {
                    RefLayout::Virtual(VirtualLayout::Runtime { pos })
                }
                ReferenceSelection::Searching => {
                    return Err(ExecutionError::new(
                        TraceError::ReferenceNotFound,
                        plan.handle_inputs,
                        plan.handle_outputs,
                    ));
                }
            };

            let mut ops = Vec::<FuseOp>::new();

            for read_ops in block_plan.reads.into_values() {
                for op in read_ops {
                    ops.push(op);
                }
            }

            for op in block.ops.iter() {
                ops.push(op.clone());
            }

            for opsw in block_plan.writes.into_values() {
                for op in opsw {
                    ops.push(op);
                }
            }

            let config = FuseBlockConfig {
                rank: plan.rank,
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
                    .as_tensor_arg(&hi.global_ir.shape.dims, hi.line_size);
                inputs.tensors.push(GlobalTensorArg::new(
                    arg,
                    hi.precision.into_elem(),
                    hi.broadcated,
                ));
            }
            HandleInput::QuantValues(hi) => {
                let arg = hi
                    .handle
                    .as_tensor_arg(&hi.global_ir.shape.dims, hi.line_size);
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
                vectorization: line_size,
                #[cfg(feature = "autotune-checks")]
                relative_id,
                ..
            } => {
                let arg = handle.as_tensor_arg(global_shape, *line_size);

                let elem = match precision {
                    FuseType::Bool => match elem_dtype::<BT>() {
                        DType::U32 => FuseType::U32.into_elem(),
                        DType::U8 => FuseType::U8.into_elem(),
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
    scalars: impl Iterator<Item = &'h (FuseType, u64)>,
    views: impl DoubleEndedIterator<Item = &'h TensorView>,
    context: &mut Context<'_, CubeFusionHandle<R>>,
    inputs: &mut GlobalArgsLaunch<'h, R>,
) {
    for (precision, id) in scalars {
        let dtype = precision.into_type();
        match context.scalars.get(&ScalarId { value: *id }) {
            Some(scalar) => match scalar {
                ScalarIr::Float(val) => inputs.scalars.push(InputScalar::new(*val, dtype)),
                ScalarIr::Int(val) => inputs.scalars.push(InputScalar::new(*val, dtype)),
                ScalarIr::UInt(val) => inputs.scalars.push(InputScalar::new(*val, dtype)),
                ScalarIr::Bool(val) => inputs.scalars.push(InputScalar::new(*val as u8, dtype)),
            },
            None => panic!("Scalar ID not found"),
        }
    }

    for relative in views {
        if let TensorView::Reshape { reshaped, .. } = relative {
            let global = context.tensors.get(reshaped).unwrap();

            for shape in global.shape.iter() {
                inputs.reshapes.push(ScalarArg::new(*shape));
            }
        }
    }
}
