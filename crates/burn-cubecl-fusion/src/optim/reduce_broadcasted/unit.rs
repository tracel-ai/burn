use crate::{
    engine::codegen::{
        self,
        ir::{FuseArg, FuseBlockConfig, GlobalArgs},
        kernel::{fuse_on_write, init_locals},
    },
    optim::reduce::args::{FusedReduceArgs, FusedReduceInput, FusedReduceOutput},
};
use cubecl::{
    Runtime,
    prelude::*,
    std::{CubeOption, CubeOptionExpand, tensor::r#virtual::VirtualTensor},
};
use cubek::reduce::{
    LineMode, ReduceInstruction, ReducePrecision,
    components::{
        global::unit::GlobalFullUnitReduce,
        instructions::{ReduceOperation, ReduceOperationConfig},
    },
    init_tensors,
};

#[derive(CubeType, CubeLaunch, Clone)]
struct ReduceFuseBlock {
    #[cube(comptime)]
    op: ReduceOperationConfig,
    #[cube(comptime)]
    config: FuseBlockConfig,
    #[cube(comptime)]
    input: FuseArg,
    #[cube(comptime)]
    output: FuseArg,
}

#[derive(CubeType, CubeLaunch, Clone)]
struct ElemwiseFuseBlock {
    #[cube(comptime)]
    config: FuseBlockConfig,
}

#[cube]
fn reduce_many<P: ReducePrecision, Out: Numeric, I: ReduceInstruction<P> + Clone>(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    reduce_axis: usize,
    idle: CubeOption<bool>,
    blocks: Sequence<ReduceFuseBlock>,
    block_end: CubeOption<ElemwiseFuseBlock>,
) {
    let global_index = ABSOLUTE_POS;
    let mut axis_size = 0;

    let mut locals = Registry::<u32, Line<Out>>::new();

    #[unroll]
    for i in 0..blocks.len() {
        let block = blocks.index(i);
        let input = FusedReduceInput {
            global: inputs.clone(),
            config: comptime!(block.config.clone()),
            arg: comptime!(block.input.clone()),
        };
        let mut output = FusedReduceOutput {
            global: outputs.clone(),
            config: comptime!(block.config.clone()),
            arg: comptime!(block.output.clone()),
        };

        let (input, mut output) = init_tensors::<FusedReduceArgs, P::EI, Out>(&input, &mut output);

        // let num_local_inputs = comptime!(block.config.local_inputs.len());
        // #[unroll]
        // for j in 0..num_local_inputs {
        //     let local_input = comptime!(block.config.local_inputs.get(j).unwrap());
        //     // Using the source locals. SHOULD BUILD A GLOBAL SHARED REGISTRY.
        //     let val = codegen::io::read::<f32>(
        //         inputs,
        //         outputs,
        //         locals,
        //         global_index,
        //         comptime!(local_input.src_arg.clone()),
        //         &block.config,
        //     );
        //     codegen::io::write::<f32>(
        //         inputs,
        //         outputs,
        //         locals,
        //         global_index,
        //         val,
        //         comptime!(local_input.dst_arg.clone()),
        //         &block.config,
        //     );
        // }

        axis_size = reduce_step::<P, Out, ReduceOperation>(
            &input,
            &mut output,
            reduce_axis,
            global_index,
            idle,
            &mut locals,
            block.op,
            comptime!(block.output.clone()),
        );
    }

    match block_end {
        CubeOption::Some(block) => {
            let width = comptime!(block.config.width as u32);
            let num_iter = axis_size / usize::cast_from(width);
            for i in 0..num_iter {
                // Register block local inputs.
                let values = Registry::<FuseArg, Line<f32>>::new();
                let args = comptime![Vec::<FuseArg>::new()];
                let index = global_index * num_iter + i;
                let mut locals = init_locals(inputs, outputs, &block.config);
                fuse_on_write::<f32>(
                    inputs,
                    outputs,
                    &mut locals,
                    index,
                    values,
                    args,
                    &block.config,
                )
            }
        }
        CubeOption::None => {}
    }
}

#[cube]
fn reduce_step<P: ReducePrecision, Out: Numeric, I: ReduceInstruction<P>>(
    input: &VirtualTensor<P::EI>,
    output: &mut VirtualTensor<Out, ReadWrite>,
    reduce_axis: usize,
    reduce_index: usize,
    idle: CubeOption<bool>,
    locals: &mut Registry<u32, Line<Out>>,
    #[comptime] config: I::Config,
    #[comptime] arg: FuseArg,
) -> usize {
    let inst = I::from_config(config);
    let axis_size = input.shape(reduce_axis);

    let acc = GlobalFullUnitReduce::reduce_single::<P, Out, I>(
        input,
        output,
        reduce_axis,
        reduce_index,
        &inst,
        idle,
        LineMode::Parallel,
    );
    let line = I::merge_line::<Out>(&inst, acc, axis_size);

    locals.insert(0u32, Line::cast_from(line));

    axis_size
}
