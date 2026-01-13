use crate::{
    engine::codegen::{
        self,
        ir::{FuseArg, FuseBlockConfig, GlobalArgs, LocalArgs},
        kernel::fuse_on_write,
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
    locals: &mut LocalArgs,
    reduce_axis: u32,
    idle: CubeOption<bool>,
    blocks: Sequence<ReduceFuseBlock>,
    block_end: CubeOption<ElemwiseFuseBlock>,
) {
    let global_index = ABSOLUTE_POS;
    let mut axis_size = 0u32;

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

        axis_size = reduce_step::<P, Out, ReduceOperation>(
            &input,
            &mut output,
            reduce_axis,
            global_index,
            idle,
            locals,
            block.op,
            comptime!(block.output.clone()),
        );
    }

    match block_end {
        CubeOption::Some(block) => {
            let width = comptime!(block.config.width as u32);
            let num_iter = axis_size / width;
            for i in 0..num_iter {
                let values = Registry::<FuseArg, Line<f32>>::new();
                let args = comptime![Sequence::<FuseArg>::new()];
                let index = global_index * num_iter + i;
                fuse_on_write::<f32>(inputs, outputs, locals, index, values, args, &block.config)
            }
        }
        CubeOption::None => {}
    }
}

#[cube]
fn reduce_step<P: ReducePrecision, Out: Numeric, I: ReduceInstruction<P>>(
    input: &VirtualTensor<P::EI>,
    output: &mut VirtualTensor<Out, ReadWrite>,
    reduce_axis: u32,
    reduce_index: u32,
    idle: CubeOption<bool>,
    locals: &mut LocalArgs,
    #[comptime] config: I::Config,
    #[comptime] arg: FuseArg,
) -> u32 {
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
    codegen::io::write_scalar::<Out>(locals, Line::cast_from(line), arg);

    axis_size
}
