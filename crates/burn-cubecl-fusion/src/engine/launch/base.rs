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
use cubecl::client::Client;
use std::marker::PhantomData;

/// The launcher is responsible to launch a fused kernel using the [TraceRunner] and a [FuseTrace].
///
/// TODO: We can reuse the same launcher between runs and avoid a lot of allocation, by simply
/// resetting the state.
pub struct FuseTraceLauncher<'a, Runner: TraceRunner> {
    trace: &'a FuseTrace,
    runner: &'a Runner,
}

impl<'a, Runner: TraceRunner> FuseTraceLauncher<'a, Runner> {
    /// Creates a new launcher.
    pub fn new(trace: &'a FuseTrace, runner: &'a Runner) -> Self {
        Self { trace, runner }
    }
    /// Launches the fuse kernel on the given device modifying the context.
    pub fn launch(
        &self,
        client: &Client,
        device: &cubecl::Device,
        context: &mut Context<CubeFusionHandle>,
    ) -> Result<TuneOutput, TraceError<Runner::Error>> {
        let mut plan = LaunchPlan::new(&self.trace.blocks);

        InputPlanner::new(&self.trace.resources, &self.trace.blocks).run(context, &mut plan);

        OutputPlanner::new(&self.trace.resources, &self.trace.blocks)
            .run(client, device, context, &mut plan);

        VectorizationPlanner::new(&self.trace.resources, &self.trace.blocks).run(
            client,
            self.runner,
            context,
            &mut plan,
        );

        match LaunchPlanExecutor::new(&self.trace.resources, &self.trace.blocks).execute::<_>(
            client,
            self.runner,
            context,
            plan,
        ) {
            Err(err) => {
                self.rollback(context, err.handles_input, err.handles_output);
                Err(err.error)
            }
            Ok(val) => Ok(val),
        }
    }

    fn rollback(
        &self,
        context: &mut Context<CubeFusionHandle>,
        handle_inputs: Vec<HandleInput>,
        handle_outputs: Vec<HandleOutput>,
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
