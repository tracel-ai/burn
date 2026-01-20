use crate::{
    engine::codegen::{
        ir::{FuseArg, FuseBlockConfig, FuseType, GlobalArgs},
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
    routines::UnitReduceBlueprint,
};

#[derive(CubeType, CubeLaunch, Clone)]
pub struct ReduceFuseBlock {
    #[cube(comptime)]
    op: ReduceOperationConfig,
    #[cube(comptime)]
    config: FuseBlockConfig,
    #[cube(comptime)]
    input: FuseArg,
    #[cube(comptime)]
    output: FuseArg,
    #[cube(comptime)]
    blueprint: UnitReduceBlueprint,
}

#[derive(CubeType, CubeLaunch, Clone)]
pub struct ElemwiseFuseBlock {
    #[cube(comptime)]
    config: FuseBlockConfig,
}

#[cube(launch_unchecked)]
pub fn reduce_br_kernel(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    reduce_axis: usize,
    blocks: Sequence<ReduceFuseBlock>,
    block_end: CubeOption<ElemwiseFuseBlock>,
) {
    reduce_many(inputs, outputs, reduce_axis, blocks, block_end);
}

const REDUCE_INPUT: u8 = 0;
const REDUCE_ACC: u8 = 1;
const REDUCE_OUT: u8 = 2;

type In = NumericExpand<REDUCE_INPUT>;
type Acc = NumericExpand<REDUCE_ACC>;
type Out = NumericExpand<REDUCE_OUT>;

#[cube]
fn set_polyfill_block(block: &ReduceFuseBlock) {
    let input_precision = comptime!(block.input.precision());
    let output_precision = comptime!(block.output.precision());
    let acc_precision = comptime!(match input_precision {
        FuseType::F64 => FuseType::F64,
        FuseType::F32 => FuseType::F32,
        FuseType::Flex32 => FuseType::F32,
        FuseType::F16 => FuseType::F32,
        FuseType::BF16 => FuseType::F32,
        FuseType::I64 => FuseType::I64,
        FuseType::I32 => FuseType::I32,
        FuseType::I16 => FuseType::I32,
        FuseType::I8 => FuseType::I32,
        FuseType::U64 => FuseType::U64,
        FuseType::U32 => FuseType::U32,
        FuseType::U16 => FuseType::U32,
        FuseType::U8 => FuseType::U32,
        FuseType::Bool => FuseType::I32,
    });

    set_polyfill::<In>(comptime!(input_precision.into_type()));
    set_polyfill::<Out>(comptime!(output_precision.into_type()));
    set_polyfill::<Acc>(comptime!(acc_precision.into_type()));
}

#[cube]
fn reduce_many(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    reduce_axis: usize,
    blocks: Sequence<ReduceFuseBlock>,
    block_end: CubeOption<ElemwiseFuseBlock>,
) {
    let mut axis_size = 0;

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

        set_polyfill_block(&block);
        let (input, mut output) = init_tensors::<FusedReduceArgs, In, Out>(&input, &mut output);

        axis_size = reduce_step::<(In, Acc), Out, ReduceOperation>(
            &input,
            &mut output,
            reduce_axis,
            block.op,
            &block.blueprint,
        );
    }

    let global_index = ABSOLUTE_POS;

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
/// Perform fuse-on-read and that's it.
fn reduce_step<P: ReducePrecision, Out: Numeric, I: ReduceInstruction<P>>(
    input: &VirtualTensor<P::EI>,
    output: &mut VirtualTensor<Out, ReadWrite>,
    reduce_axis: usize,
    #[comptime] config: I::Config,
    #[comptime] blueprint: &UnitReduceBlueprint,
) -> usize {
    let inst = I::from_config(config);
    let axis_size = input.shape(reduce_axis);

    GlobalFullUnitReduce::execute::<P, Out, I>(
        input,
        output,
        reduce_axis,
        &inst,
        LineMode::Parallel,
        comptime!(blueprint.clone()),
    );
    axis_size
}
