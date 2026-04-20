use super::{HandleInput, HandleOutput, LaunchPlan, ReferenceSelection};
use crate::engine::launch::runner::TraceRunner;
use crate::engine::trace::{FuseResources, TensorView, TraceError, TuneOutput, block::FuseBlock};
use crate::{
    CubeFusionHandle,
    engine::{
        codegen::ir::{
            FuseBlockConfig, FuseOp, FuseType, GlobalArgsLaunch, RefLayout, VirtualLayout,
        },
        codegen::tensor::GlobalTensorArg,
    },
};
use burn_fusion::stream::{Context, ScalarId};
use burn_ir::ScalarIr;
use cubecl::{
    Runtime,
    client::ComputeClient,
    ir::{AddressType, Type},
    prelude::{InputScalar, TensorArg},
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

    pub fn execute<Runner: TraceRunner<R>>(
        self,
        client: &ComputeClient<R>,
        runner: &Runner,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        plan: LaunchPlan<'a, R>,
    ) -> Result<TuneOutput<R>, ExecutionError<R, Runner>> {
        let mut num_writes = 0;
        for b in plan.blocks.iter() {
            for writes in b.writes.values() {
                num_writes += writes.len();
            }
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

        register_inputs(plan.handle_inputs.clone(), &mut inputs);
        register_scalars(
            self.resources.scalars.iter(),
            self.resources.views.iter(),
            context,
            &mut inputs,
        );
        register_outputs::<R>(plan.handle_outputs.clone(), &mut outputs, &mut tune_output);

        for layout in plan.runtime_layouts {
            for s in layout.shape.iter() {
                inputs.runtime_layouts.push(*s);
            }
            for s in layout.strides.iter() {
                inputs.runtime_layouts.push(*s);
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
                        vector_size: block_plan.width,
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

fn register_inputs<R: Runtime>(
    handle_inputs: Vec<HandleInput<R>>,
    inputs: &mut GlobalArgsLaunch<R>,
) {
    for hi in handle_inputs {
        match hi {
            HandleInput::Normal(hi) => {
                let at = hi.handle.required_address_type();
                let arg = hi.handle.into_tensor_arg(hi.global_ir.shape.clone());
                inputs.tensors.push(GlobalTensorArg::new(
                    arg,
                    hi.precision.into_type(hi.vector_size),
                    hi.broadcated,
                    at,
                ));
            }
            HandleInput::QuantValues(hi) => {
                let at = hi.handle.required_address_type();
                let arg = hi.handle.into_tensor_arg(hi.global_ir.shape.clone());
                inputs.tensors.push(GlobalTensorArg::new(
                    arg,
                    hi.precision.into_type(hi.vector_size),
                    false,
                    at,
                ));
            }
            HandleInput::QuantParams(hi) => {
                let at = hi.handle.required_address_type();
                let arg = hi.handle.into_tensor_arg(hi.shape.clone());
                inputs.tensors.push(GlobalTensorArg::new(
                    arg,
                    hi.precision.into_type(1),
                    false,
                    at,
                ));
            }
        }
    }
}

fn register_outputs<R: Runtime>(
    handle_outputs: Vec<HandleOutput<R>>,
    outputs: &mut GlobalArgsLaunch<R>,
    #[allow(unused_variables)] tune_output: &mut TuneOutput<R>,
) {
    for item in handle_outputs {
        match item {
            HandleOutput::Alias {
                input_pos,
                precision,
                global_shape,
                strides,
                #[cfg(feature = "autotune-checks")]
                debug_info,
            } => {
                outputs.tensors.push(GlobalTensorArg::new(
                    TensorArg::Alias {
                        input_pos,
                        strides,
                        shape: global_shape,
                    },
                    precision.into_type(1),
                    false,
                    AddressType::default(),
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
                vectorization: vector_size,
                #[cfg(feature = "autotune-checks")]
                relative_id,
                ..
            } => {
                let at = handle.required_address_type();

                #[cfg(feature = "autotune-checks")]
                if let TuneOutput::Checked { handles, .. } = tune_output {
                    handles.insert(relative_id, (global_shape.clone(), handle.clone()));
                }

                let arg = handle.into_tensor_arg(global_shape.clone());

                let elem = precision.into_elem();
                let ty = Type::new(elem.into()).with_vector_size(vector_size);

                outputs
                    .tensors
                    .push(GlobalTensorArg::new(arg, ty, false, at));
            }
        }
    }
}

fn register_scalars<'h, R: Runtime>(
    scalars: impl Iterator<Item = &'h (FuseType, u64)>,
    views: impl DoubleEndedIterator<Item = &'h TensorView>,
    context: &mut Context<'_, CubeFusionHandle<R>>,
    inputs: &mut GlobalArgsLaunch<R>,
) {
    for (precision, id) in scalars {
        let dtype = precision.into_storage_type();
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
                inputs.reshapes.push(*shape);
            }
        }
    }
}
