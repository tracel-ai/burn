use crate::{
    engine::{
        codegen::ir::FuseType,
        fuser::TraceOperationFuser,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
    },
    optim::{
        reduce::{FusedReduce, ReduceInstruction},
        reduce_broadcasted::{
            ReduceBrInfo,
            fuser::{
                block::{ReduceBlockFuser, ReduceBlockKind},
                full_analyzer::FullFuserAnalyzer,
            },
            launch::ReduceBrFuseBlock,
        },
    },
};
use burn_fusion::OperationFuser;
use cubecl::Runtime;
use cubek::reduce::components::instructions::ReduceOperationConfig;

/// Responsible for fusing a single trace for all operations involved in this optimization.
pub struct ReduceBroadcastedFullFuser {
    pub(crate) fuser: TraceOperationFuser,
    analyzer: FullFuserAnalyzer,
    blocks: Vec<ReduceBlockKind>,
    settings_read: FuseSettings,
    settings_write: FuseSettings,
}

impl ReduceBroadcastedFullFuser {
    /// Creates a new fuser with the given settings.
    pub fn new(max_bindings: u32, bool_precision: FuseType, analyzer: FullFuserAnalyzer) -> Self {
        let settings_read = FuseSettings {
            inplace: false,
            ref_layout: RefLayoutSetting::OnlyContiguous,
            // TODO: Only for debuging for now
            vectorization: VectorizationSetting::Activated,
            ..Default::default()
        };
        let settings_write = FuseSettings {
            output_shape_updates: false,
            ref_layout: RefLayoutSetting::OnlyContiguous,
            // TODO: Fusion axis should be on the (reduce_axis - 1).
            vectorization: VectorizationSetting::Deactivated,
            ..Default::default()
        };
        let fuser = TraceOperationFuser::new(max_bindings, bool_precision, settings_read);

        Self {
            fuser,
            blocks: Vec::new(),
            settings_write,
            settings_read,
            analyzer,
        }
    }

    pub fn finish(mut self) -> ReduceBrInfo {
        let mut reduce_axis = 0;
        let mut blocks = Vec::new();

        for block in self.blocks.iter() {
            match block {
                ReduceBlockKind::Elemwise => {}
                ReduceBlockKind::Reduce { reduce, .. } => {
                    let config = match reduce.inst {
                        ReduceInstruction::ArgMax => ReduceOperationConfig::ArgMax,
                        ReduceInstruction::ArgMin => ReduceOperationConfig::ArgMin,
                        ReduceInstruction::Prod => ReduceOperationConfig::Prod,
                        ReduceInstruction::Mean => ReduceOperationConfig::Mean,
                        ReduceInstruction::Sum => ReduceOperationConfig::Sum,
                        ReduceInstruction::Max => ReduceOperationConfig::Max,
                        ReduceInstruction::Min => ReduceOperationConfig::Min,
                        ReduceInstruction::MaxAbs => ReduceOperationConfig::MaxAbs,
                    };

                    let info = ReduceBrFuseBlock {
                        op: config,
                        input: reduce.input.clone(),
                        output: reduce.output.clone(),
                    };
                    reduce_axis = reduce.axis;
                    blocks.push(info);
                }
            }
        }

        let trace = self.fuser.finish();

        ReduceBrInfo {
            blocks,
            trace,
            reduce_axis,
        }
    }

    /// Registers a [ReduceBlockFuser] to build the trace.
    pub fn register<R: Runtime>(&mut self, block: &ReduceBlockFuser<R>) {
        // Helper to close previous blocks if necessary
        if !self.fuser.is_empty() {
            let mut settings = self.settings_read;
            settings.vectorization = VectorizationSetting::EqualThanPreviousBlock { block_pos: 0 };
            settings.ref_layout = RefLayoutSetting::SameAsBlock { block_pos: 0 };
            self.fuser.next_block([], settings);

            let analysis = self.analyzer.retrieve_next();

            for (tensor, block_pos) in analysis.inputs {
                self.fuser.block_local_input(&tensor, block_pos);
            }
        }

        match &block.kind {
            ReduceBlockKind::Elemwise => {
                for op in &block.ops {
                    self.fuser.fuse(op);
                }
                self.blocks.push(ReduceBlockKind::Elemwise);
            }
            ReduceBlockKind::Reduce { ops_index, reduce } => {
                for op in &block.ops[0..*ops_index] {
                    self.fuser.fuse(op);
                }

                let [input] = self
                    .fuser
                    .next_block([&reduce.op.input], self.settings_write);
                println!("Thing to work with as reduce input: {input:?}");

                let output = self.fuser.output_unhandled(&reduce.op.out);
                let analysis = self.analyzer.retrieve_next();

                println!("Analysis {:?} op: {:?}", analysis.inputs, reduce.op);
                for (tensor, block_pos) in analysis.inputs {
                    self.fuser.block_local_input(&tensor, block_pos);
                }

                let fused_reduce = FusedReduce {
                    input,
                    output,
                    acc: reduce.acc,
                    axis: reduce.axis,
                    op: reduce.op.clone(),
                    use_planes: reduce.use_planes,
                    shared: reduce.shared,
                    inst: reduce.inst,
                };

                self.blocks.push(ReduceBlockKind::Reduce {
                    ops_index: *ops_index,
                    reduce: fused_reduce,
                });

                for op in &block.ops[*ops_index + 1..block.ops.len()] {
                    self.fuser.fuse(op);
                }
            }
        }
    }
}
