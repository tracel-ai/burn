use crate::{
    engine::{
        codegen::ir::{FuseArg, FuseBlockConfig, GlobalArgsLaunch},
        launch::runner::{TraceRunner, Vectorization},
    },
    optim::reduce_broadcasted::unit::{
        ElemwiseFuseBlockLaunch, ReduceFuseBlockLaunch, reduce_br_kernel,
    },
};
use cubecl::{
    Runtime,
    ir::{ElemType, FloatKind, StorageType},
    prelude::{ComputeClient, ScalarArg, SequenceArg},
    server::LaunchError,
    std::CubeOptionArgs,
};
use cubek::reduce::{
    LineMode, ReduceDtypes,
    components::instructions::ReduceOperationConfig,
    launch::RoutineStrategy,
    routines::{
        BlueprintStrategy, GlobalReduceBlueprint, ReduceLineSettings, ReduceProblem, Routine,
        unit::{UnitRoutine, UnitStrategy},
    },
};

pub struct ReduceBrFuseBlock {
    op: ReduceOperationConfig,
    input: FuseArg,
    output: FuseArg,
}

#[derive(new)]
pub struct FusedReduceBroadcastedLaunch<'a> {
    blocks: &'a Vec<ReduceBrFuseBlock>,
    reduce_axis: usize,
    strategy: RoutineStrategy,
}

impl<R: Runtime> Vectorization<R> for FusedReduceBroadcastedLaunch<'_> {}

impl<R: Runtime> TraceRunner<R> for FusedReduceBroadcastedLaunch<'_> {
    type Error = LaunchError;

    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        configs: &'a [FuseBlockConfig],
    ) -> Result<(), Self::Error> {
        let routine = UnitRoutine;
        let first_config = &configs[0];
        let shape = inputs.shape_ref(&first_config.ref_layout, first_config.rank);

        let vector_size = shape[self.reduce_axis];
        let vector_count = shape.iter().product::<usize>() / vector_size;

        let (blueprint, settings) = routine
            .prepare::<R>(
                client,
                ReduceProblem {
                    vector_size,
                    vector_count,
                    axis: self.reduce_axis,
                    dtypes: ReduceDtypes {
                        input: StorageType::Scalar(ElemType::Float(FloatKind::F32)),
                        output: StorageType::Scalar(ElemType::Float(FloatKind::F32)),
                        accumulation: StorageType::Scalar(ElemType::Float(FloatKind::F32)),
                    },
                },
                ReduceLineSettings {
                    line_mode: LineMode::Parallel,
                    line_size_input: first_config.width,
                    line_size_output: 1,
                },
                BlueprintStrategy::Inferred(UnitStrategy),
            )
            .unwrap();

        let mut blocks = SequenceArg::new();

        self.blocks
            .iter()
            .zip(configs.iter())
            .for_each(|(block, config)| {
                let arg = ReduceFuseBlockLaunch::new(
                    block.op,
                    config.clone(),
                    block.input.clone(),
                    block.output.clone(),
                    match blueprint.global {
                        GlobalReduceBlueprint::Unit(bpt) => bpt,
                        _ => panic!(),
                    },
                );
                blocks.push(arg);
            });

        let block_end = match configs.len() > self.blocks.len() {
            true => CubeOptionArgs::Some(ElemwiseFuseBlockLaunch::new(
                configs.last().cloned().unwrap(),
            )),
            false => CubeOptionArgs::None,
        };

        unsafe {
            reduce_br_kernel::launch_unchecked::<R>(
                client,
                settings.cube_count,
                settings.cube_dim,
                inputs,
                outputs,
                ScalarArg::new(self.reduce_axis),
                blocks,
                block_end,
            )?;
        }

        Ok(())
    }
}
