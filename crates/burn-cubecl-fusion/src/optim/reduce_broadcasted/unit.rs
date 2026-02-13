use crate::{
    engine::codegen::{
        ir::{FuseArg, FuseBlockConfig, FuseType, GlobalArgs, multi_block_variables_init},
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

/// A configuration block for a reduction operation within a fused kernel.
///
/// This struct holds all the compile-time information needed to perform a
/// reduction, including the operation type (Sum, Max, etc.) and the layout
/// configuration for both input and output.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct ReduceFuseBlock {
    #[cube(comptime)]
    op: ReduceOperationConfig,
    #[cube(comptime)]
    config_input: FuseBlockConfig,
    #[cube(comptime)]
    config_output: FuseBlockConfig,
    #[cube(comptime)]
    input: FuseArg,
    #[cube(comptime)]
    output: FuseArg,
    #[cube(comptime)]
    blueprint: UnitReduceBlueprint,
}

/// A configuration block for an elementwise operation that follows a reduction.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct ElemwiseFuseBlock {
    #[cube(comptime)]
    config: FuseBlockConfig,
}

/// The entry point for a broadcasted reduction kernel.
///
/// This kernel initializes local variables for multiple reduction blocks and then
/// executes the reduction sequence.
///
/// # Arguments
///
/// * `inputs` - Global arguments containing input tensor handles.
/// * `outputs` - Global arguments containing output tensor handles.
/// * `reduce_axis` - The dimension along which the reduction is performed.
/// * `blocks` - A sequence of reduction operations to execute.
/// * `block_end` - An optional elementwise block to execute after reductions are complete.
#[cube(launch_unchecked, address_type = "dynamic")]
pub fn reduce_kernel_broadcasted(
    inputs: &GlobalArgs,
    outputs: &mut GlobalArgs,
    reduce_axis: usize,
    blocks: Sequence<ReduceFuseBlock>,
    block_end: CubeOption<ElemwiseFuseBlock>,
) {
    #[unroll]
    for i in 0..blocks.len() {
        let block = blocks.index(i);
        multi_block_variables_init(&block.config_input, &mut outputs.variables);
        multi_block_variables_init(&block.config_output, &mut outputs.variables);
    }

    reduce_many(inputs, outputs, reduce_axis, blocks, block_end);
}

const REDUCE_INPUT: u8 = 0;
const REDUCE_ACC: u8 = 1;
const REDUCE_OUT: u8 = 2;

type In = NumericExpand<REDUCE_INPUT>;
type Acc = NumericExpand<REDUCE_ACC>;
type Out = NumericExpand<REDUCE_OUT>;

/// Configures the precision polyfills for the reduction based on the block's `FuseType`.
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

/// Internal logic for executing a sequence of reduction blocks followed by an optional
/// trailing elementwise block.
#[cube]
#[allow(clippy::clone_on_copy)]
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
            config: comptime!(block.config_input.clone()),
            arg: comptime!(block.input.clone()),
        };
        let global = outputs.clone();
        let config = comptime!(block.config_output.clone());
        let arg = comptime!(block.output.clone());
        let mut output = FusedReduceOutput {
            global,
            config,
            arg,
        };

        set_polyfill_block(block);
        let (input, mut output) = init_tensors::<FusedReduceArgs, In, Out>(&input, &mut output);

        axis_size = reduce_step::<(In, Acc), Out, ReduceOperation>(
            &input,
            &mut output,
            reduce_axis,
            block.op,
            comptime!(block.blueprint.clone()),
        );
    }

    match block_end {
        CubeOption::Some(block) => {
            let global_index = ABSOLUTE_POS;
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
                    &block.config.clone(),
                )
            }
        }
        CubeOption::None => {}
    }
}

#[cube]
/// Executes a single reduction step using a specified instruction and blueprint.
///
/// Returns the size of the axis that was reduced.
fn reduce_step<P: ReducePrecision, Out: Numeric, I: ReduceInstruction<P>>(
    input: &VirtualTensor<P::EI>,
    output: &mut VirtualTensor<Out, ReadWrite>,
    reduce_axis: usize,
    #[comptime] config: I::Config,
    #[comptime] blueprint: UnitReduceBlueprint,
) -> usize {
    let inst = I::from_config(config);
    let axis_size = input.shape(reduce_axis);

    GlobalFullUnitReduce::execute::<P, Out, I>(
        input,
        output,
        reduce_axis,
        &inst,
        LineMode::Parallel,
        comptime!(blueprint),
    );
    axis_size
}
