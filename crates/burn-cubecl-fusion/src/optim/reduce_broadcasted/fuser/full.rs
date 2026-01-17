use crate::{
    engine::{
        codegen::ir::FuseType,
        fuser::TraceOperationFuser,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
    },
    optim::{
        reduce::FusedReduce,
        reduce_broadcasted::fuser::{
            block::{ReduceBlockFuser, ReduceBlockKind},
            full_analyzer::FullFuserAnalyzer,
        },
    },
};
use burn_fusion::OperationFuser;
use cubecl::Runtime;

/// Responsible for fusing a single trace for all operations involved in this optimization.
pub struct ReduceBroadcastedFullFuser {
    pub(crate) fuser: TraceOperationFuser,
    analyzer: FullFuserAnalyzer,
    blocks: Vec<ReduceBlockKind>,
    settings_write: FuseSettings,
}

impl ReduceBroadcastedFullFuser {
    /// Creates a new fuser with the given settings.
    pub fn new(max_bindings: u32, bool_precision: FuseType, analyzer: FullFuserAnalyzer) -> Self {
        let fuser = TraceOperationFuser::new(
            max_bindings,
            bool_precision,
            FuseSettings {
                inplace: false,
                ref_layout: RefLayoutSetting::OnlyContiguous,
                ..Default::default()
            },
        );
        let settings_write = FuseSettings {
            output_shape_updates: false,
            // TODO: Fusion axis should be on the (reduce_axis - 1).
            vectorization: VectorizationSetting::SmallerOrEqualThanPreviousBlock,
            ..Default::default()
        };

        Self {
            fuser,
            blocks: Vec::new(),
            settings_write,
            analyzer,
        }
    }

    /// Registers a [ReduceBlockFuser] to build the trace.
    pub fn register<R: Runtime>(&mut self, block: &ReduceBlockFuser<R>) {
        println!("Registering ...");
        // Helper to close previous blocks if necessary
        if !self.fuser.is_empty() {
            self.fuser.next_block([], self.settings_write).unwrap();

            let analysis = self.analyzer.retrieve_next();

            for (tensor, block_pos) in analysis.inputs {
                self.fuser.block_local_input(&tensor, block_pos);
            }
        }

        match &block.kind {
            ReduceBlockKind::Elemwise => {
                for op in &block.ops {
                    println!("Registering elemwise {:?}", op);
                    self.fuser.fuse(op);
                }
                self.blocks.push(ReduceBlockKind::Elemwise);
            }
            ReduceBlockKind::Reduce { ops_index, reduce } => {
                for op in &block.ops[0..*ops_index] {
                    println!("Registering fuse-on-read {:?}", op);
                    self.fuser.fuse(op);
                }

                let [input] = self
                    .fuser
                    .next_block([&reduce.op.input], self.settings_write)
                    .unwrap();

                let output = self.fuser.output_unhandled(&reduce.op.out);
                let analysis = self.analyzer.retrieve_next();

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
                    println!("Registering fuse-on-write {:?}", op);
                    self.fuser.fuse(op);
                }
            }
        }
    }
}
