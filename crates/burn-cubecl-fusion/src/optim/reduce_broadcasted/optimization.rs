use crate::{
    CubeFusionHandle, FallbackOperation,
    engine::{
        codegen::{
            self,
            ir::{FuseArg, FuseBlockConfig, GlobalArgs, LocalArgs},
            kernel::fuse_on_write,
        },
        trace::{TraceError, TuneOutput},
    },
    optim::{
        elemwise::ElemwiseOptimization,
        reduce::{
            FusedReduceError, ReduceOptimizationInfo, ReduceOptimizationState,
            ReduceOptimizationTuneArg,
            args::{FusedReduceArgs, FusedReduceInput, FusedReduceOutput},
        },
        reduce_broadcasted::tune::fused_broadcasted_reduce_autotune,
    },
};
use burn_fusion::stream::Context;
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
    launch::RoutineStrategy,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub struct ReduceBroadcastedOptimization<R: Runtime> {
    pub(crate) info: Arc<ReduceBroadcastedOptimizationInfo<R>>,
    pub(crate) num_ops: usize,
}

pub(crate) struct ReduceBroadcastedOptimizationInfo<R: Runtime> {
    pub(crate) fallbacks: Vec<ReduceBlockOptimInfo<R>>,
}

pub(crate) enum ReduceBlockOptimInfo<R: Runtime> {
    Reduce(Arc<ReduceOptimizationInfo<R>>),
    Elemwise(Arc<ElemwiseOptimization<R>>),
}

pub(crate) struct ReduceBroadcastedOptimizationTuneArg<R: Runtime> {
    pub(crate) fallbacks: Vec<ReduceBlockOptimArg<R>>,
    pub(crate) client: ComputeClient<R>,
    pub(crate) device: R::Device,
}

pub(crate) enum ReduceBlockOptimArg<R: Runtime> {
    Reduce(ReduceOptimizationTuneArg<R>),
    Elemwise(Arc<ElemwiseOptimization<R>>),
}

impl<R: Runtime> ReduceBlockOptimArg<R> {
    pub fn execute_fallback<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> Option<TuneOutput<R>> {
        match self {
            ReduceBlockOptimArg::Reduce(reduce) => Some(reduce.execute_fallback::<BT>(context)),
            ReduceBlockOptimArg::Elemwise(elem) => {
                elem.execute::<BT>(context);
                None
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReduceBroadcastedOptimizationState {
    fallbacks: Vec<ReduceOptimizationState>,
}

impl<R: Runtime> ReduceBroadcastedOptimizationTuneArg<R> {
    pub fn execute_fused<BT: CubeElement>(
        &self,
        _context: &mut Context<'_, CubeFusionHandle<R>>,
        _strategy: RoutineStrategy,
    ) -> Result<TuneOutput<R>, TraceError<FusedReduceError>> {
        todo!()
    }

    pub fn execute_fallback<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) {
        println!("execute N fallbacks: {}", self.fallbacks.len());
        for fallback in self.fallbacks.iter() {
            fallback.execute_fallback::<BT>(context);
        }
    }
}

#[allow(clippy::too_many_arguments)]
impl<R: Runtime> ReduceBroadcastedOptimization<R> {
    /// Execute the optimization.
    pub fn execute<BT: CubeElement>(
        &mut self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        fallback: impl Fn(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        let mut current_index = 0;
        let mut client = None;
        let mut device = None;

        let fallbacks = self
            .info
            .fallbacks
            .iter()
            .map(|info| {
                match info {
                    ReduceBlockOptimInfo::Reduce(info) => {
                        // The index of the fallback reduce is the number of ops fused as read.
                        let fallback = fallback(current_index + info.len_read);
                        client = Some(info.client.clone());
                        device = Some(info.device.clone());
                        let arg = ReduceOptimizationTuneArg {
                            info: info.clone(),
                            fallback,
                        };
                        current_index += info.len;
                        ReduceBlockOptimArg::Reduce(arg)
                    }
                    ReduceBlockOptimInfo::Elemwise(op) => ReduceBlockOptimArg::Elemwise(op.clone()),
                }
            })
            .collect();

        let arg = ReduceBroadcastedOptimizationTuneArg {
            fallbacks,
            client: client.unwrap(),
            device: device.unwrap(),
        };

        #[cfg(feature = "autotune")]
        fused_broadcasted_reduce_autotune::<R, BT>(arg, context);

        #[cfg(not(feature = "autotune"))]
        arg.execute_fallback::<BT>(context);
    }

    // pub fn num_output_buffers(&self) -> usize {
    //     todo!()
    // }

    pub fn to_state(&self) -> ReduceBroadcastedOptimizationState {
        todo!()
    }

    pub fn from_state(_device: &R::Device, _state: ReduceBroadcastedOptimizationState) -> Self {
        todo!()
    }

    /// Returns the number of output buffers added by fusion.
    pub fn num_ops_fused(&self) -> usize {
        self.num_ops
    }
}

// #[derive(CubeType, CubeLaunch, Clone)]
// struct ReduceFuseBlock {
//     #[cube(comptime)]
//     op: ReduceOperationConfig,
//     #[cube(comptime)]
//     config: FuseBlockConfig,
//     #[cube(comptime)]
//     input: FuseArg,
//     #[cube(comptime)]
//     output: FuseArg,
// }
//
// #[derive(CubeType, CubeLaunch, Clone)]
// struct ElemwiseFuseBlock {
//     #[cube(comptime)]
//     config: FuseBlockConfig,
// }
//
// #[cube]
// fn reduce_many<P: ReducePrecision, Out: Numeric, I: ReduceInstruction<P> + Clone>(
//     inputs: &GlobalArgs,
//     outputs: &mut GlobalArgs,
//     locals: &mut LocalArgs,
//     reduce_axis: u32,
//     idle: CubeOption<bool>,
//     blocks: Sequence<ReduceFuseBlock>,
//     block_end: CubeOption<ElemwiseFuseBlock>,
// ) {
//     let global_index = ABSOLUTE_POS;
//     let mut epilog = EpilogInfo {
//         axis_size: 0,
//         num_reduce: 0,
//     };
//
//     #[unroll]
//     for i in 0..blocks.len() {
//         let block = blocks.index(i);
//         let input = FusedReduceInput {
//             global: inputs.clone(),
//             config: comptime!(block.config.clone()),
//             arg: comptime!(block.input.clone()),
//         };
//         let mut output = FusedReduceOutput {
//             global: outputs.clone(),
//             config: comptime!(block.config.clone()),
//             arg: comptime!(block.output.clone()),
//         };
//
//         let (input, mut output) = init_tensors::<FusedReduceArgs, P::EI, Out>(&input, &mut output);
//
//         let updated = reduce_step::<P, Out, ReduceOperation>(
//             &input,
//             &mut output,
//             reduce_axis,
//             global_index,
//             idle,
//             locals,
//             block.op,
//             comptime!(block.output.clone()),
//         );
//         epilog.num_reduce = updated.num_reduce;
//         epilog.axis_size = updated.axis_size;
//     }
//
//     match block_end {
//         CubeOption::Some(block) => {
//             let width = block.config.width as u32;
//             let reduce_index_start = global_index * width;
//
//             for b in 0..width {
//                 let reduce_index = global_index * width;
//                 for i in 0..epilog.axis_size {
//                     let values = Registry::<FuseArg, Line<f32>>::new();
//                     let args = comptime![Sequence::<FuseArg>::new()];
//                     let index = reduce_index * epilog.axis_size + i;
//                     fuse_on_write::<f32>(
//                         inputs,
//                         outputs,
//                         locals,
//                         index,
//                         values,
//                         args,
//                         &block.config,
//                     )
//                 }
//             }
//         }
//         CubeOption::None => {}
//     }
// }
//
// #[derive(CubeType)]
// struct EpilogInfo {
//     #[cube(comptime)]
//     num_reduce: u32,
//     axis_size: u32,
// }
//
// #[cube]
// fn reduce_step<P: ReducePrecision, Out: Numeric, I: ReduceInstruction<P>>(
//     input: &VirtualTensor<P::EI>,
//     output: &mut VirtualTensor<Out, ReadWrite>,
//     reduce_axis: u32,
//     global_index: u32,
//     idle: CubeOption<bool>,
//     locals: &mut LocalArgs,
//     #[comptime] config: I::Config,
//     #[comptime] arg: FuseArg,
// ) -> EpilogInfo {
//     let inst = I::from_config(config);
//     let axis_size = input.shape(reduce_axis);
//     let line_size_output = output.line_size();
//
//     let mut buffer = Line::<Out>::empty(line_size_output);
//     let reduce_index_start = global_index * line_size_output;
//
//     #[unroll]
//     for i in 0..line_size_output {
//         let reduce_index = reduce_index_start + i;
//         let acc = GlobalFullUnitReduce::reduce_single::<P, Out, I>(
//             input,
//             output,
//             reduce_axis,
//             reduce_index,
//             &inst,
//             idle,
//             LineMode::Parallel,
//         );
//         let line = I::merge_line::<Out>(&inst, acc, axis_size);
//         buffer[i] = line;
//     }
//
//     codegen::io::write_scalar(locals, buffer, arg);
//     EpilogInfo {
//         num_reduce: line_size_output,
//         axis_size,
//     }
// }
