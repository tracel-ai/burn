use crate::{
    CubeFusionHandle,
    engine::{
        launch::{
            HandleInput, HandleOutput, LaunchPlan, executor::LaunchPlanExecutor,
            input::InputPlanner, output::OutputPlanner, runner::TraceRunner,
            vectorization::VectorizationPlanner,
        },
        trace::{FuseTrace, TraceError, TuneOutput},
    },
};
use burn_fusion::stream::Context;
use cubecl::{CubeElement, Runtime, client::ComputeClient};
use std::marker::PhantomData;

/// The launcher is responsible to launch a fused kernel using the [TraceRunner] and a [FuseTrace].
pub struct FuseTraceLauncher<'a, R: Runtime, Runner: TraceRunner<R>> {
    trace: &'a FuseTrace,
    runner: &'a Runner,
    _runtime: PhantomData<R>,
}

impl<'a, R: Runtime, Runner: TraceRunner<R>> FuseTraceLauncher<'a, R, Runner> {
    /// Creates a new launcher.
    pub fn new(trace: &'a FuseTrace, runner: &'a Runner) -> Self {
        Self {
            trace,
            runner,
            _runtime: PhantomData,
        }
    }
    /// Launches the fuse kernel on the given device modifying the context.
    pub fn launch<BT: CubeElement>(
        &self,
        client: &ComputeClient<R>,
        device: &R::Device,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> Result<TuneOutput<R>, TraceError<Runner::Error>> {
        let mut plan = LaunchPlan::new(&self.trace.blocks);

        InputPlanner::new(&self.trace.resources, &self.trace.blocks).run(context, &mut plan);

        OutputPlanner::new(&self.trace.resources, &self.trace.blocks)
            .run::<BT>(client, device, context, &mut plan);

        VectorizationPlanner::new(&self.trace.resources, &self.trace.blocks).run(
            client,
            self.runner,
            context,
            &mut plan,
        );

        match LaunchPlanExecutor::new(&self.trace.resources, &self.trace.blocks)
            .execute::<_, BT>(client, self.runner, context, plan)
        {
            Err(err) => {
                self.rollback(context, err.handles_input, err.handles_output);
                Err(err.error)
            }
            Ok(val) => Ok(val),
        }
    }

    fn rollback(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        handle_inputs: Vec<HandleInput<R>>,
        handle_outputs: Vec<HandleOutput<R>>,
    ) {
        for input in handle_inputs {
            match input {
                HandleInput::Normal(input) => {
                    context
                        .handles
                        .register_handle(input.global_ir.id, input.handle_rollback());
                }
                HandleInput::QuantValues(input) => {
                    context
                        .handles
                        .register_handle(input.global_ir.id, input.handle);
                }
                HandleInput::QuantParams(_) => {
                    // The scales are part of the quant data handle.
                }
            };
        }
        for output in handle_outputs {
            if let HandleOutput::Owned {
                global_id, handle, ..
            } = output
            {
                context.handles.register_handle(global_id, handle);
            }
        }
    }
}
