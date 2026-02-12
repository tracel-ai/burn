use crate::{
    engine::{
        codegen::ir::{FuseArg, FuseBlockConfig, GlobalArgsLaunch, RefLayout},
        launch::runner::{TraceRunner, Vectorization},
    },
    optim::reduce_broadcasted::unit::{
        ElemwiseFuseBlockLaunch, ReduceFuseBlockLaunch, reduce_kernel_broadcasted,
    },
};
use cubecl::{
    Runtime,
    ir::{AddressType, ElemType, FloatKind, StorageType},
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
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReduceBroadcastedFuseBlock {
    pub(crate) op: ReduceOperationConfig,
    pub(crate) input: FuseArg,
    pub(crate) output: FuseArg,
}

#[derive(new)]
pub struct FusedReduceBroadcastedLaunch<'a> {
    blocks: &'a Vec<ReduceBroadcastedFuseBlock>,
    reduce_axis: usize,
    // TODO: Support multiple strategies.
    _strategy: RoutineStrategy,
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

        let shape = match &first_config.ref_layout {
            RefLayout::Concrete(FuseArg::Output(..)) => {
                outputs.shape_ref(&first_config.ref_layout, first_config.rank)
            }
            _ => inputs.shape_ref(&first_config.ref_layout, first_config.rank),
        };

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
                    address_type: AddressType::default(),
                },
                ReduceLineSettings {
                    line_mode: LineMode::Parallel,
                    line_size_input: first_config.width,
                    line_size_output: 1,
                },
                BlueprintStrategy::Inferred(UnitStrategy),
            )
            .unwrap();

        assert_eq!(blueprint.line_mode, LineMode::Parallel);

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

        let block_end = match configs.len() > index {
            true => CubeOptionArgs::Some(ElemwiseFuseBlockLaunch::new(
                configs.last().cloned().unwrap(),
            )),
            false => CubeOptionArgs::None,
        };

        // TODO: Ensure parallel is selected.

        unsafe {
            reduce_kernel_broadcasted::launch_unchecked::<R>(
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
