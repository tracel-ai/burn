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

#[derive(Debug)]
pub struct ReduceBrFuseBlock {
    pub(crate) op: ReduceOperationConfig,
    pub(crate) input: FuseArg,
    pub(crate) output: FuseArg,
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
        println!("CCCCCCCCCCCCCCCC {configs:?}");
        println!("BbbbbbbbbbbbbbbBBBBBBBBBBBBBBB {:?}", self.blocks);
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

        println!("HERE");
        let mut blocks = SequenceArg::new();

        let mut index = 0;
        for block in self.blocks {
            let arg = ReduceFuseBlockLaunch::new(
                block.op,
                configs[index].clone(),
                configs[index + 1].clone(),
                block.input.clone(),
                block.output.clone(),
                match blueprint.global {
                    GlobalReduceBlueprint::Unit(bpt) => bpt,
                    _ => panic!(),
                },
            );
            index += 2;
            blocks.push(arg);
        }

        println!("THERE");
        let block_end = match configs.len() > index {
            true => CubeOptionArgs::Some(ElemwiseFuseBlockLaunch::new(
                configs.last().cloned().unwrap(),
            )),
            false => CubeOptionArgs::None,
        };
        println!("BEFORE");

        unsafe {
            let val = reduce_br_kernel::launch_unchecked::<R>(
                client,
                settings.cube_count,
                settings.cube_dim,
                inputs,
                outputs,
                ScalarArg::new(self.reduce_axis),
                blocks,
                block_end,
            );
            panic!("{val:?}");
        }

        Ok(())
    }
}
