use crate::{
    engine::{
        codegen::ir::FuseType,
        fuser::TraceOperationFuser,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
    },
    optim::{
        CubeOptimization,
        elemwise::ElemwiseOptimization,
        reduce::{FusedReduce, ReduceFuser, ReduceFuserInfo},
        reduce_broadcasted::ReduceBlockOptimInfo,
    },
};
use burn_fusion::{FuserProperties, OperationFuser};
use burn_ir::OperationIr;
use cubecl::Runtime;
use std::sync::Arc;

/// This type is responsible to fuse a single reduce block or elemwise block.
///
/// When the kind of this block is a reduce, the block supports for fuse-on-read and
/// fuse-on-write fusion.
///
/// Broadcasting isn't supported, and another block should fuse it instead.
pub struct ReduceBlockFuser<R: Runtime> {
    /// We're using the [ReduceFuser] for both elemwise and reduce blocks where we only keep the
    /// fuse-on-read trace if the block is tagged as elemwise.
    ///
    /// # Notes
    ///
    /// We can only have a single elemwise block at the end in a full [ReduceBlockFuser] at the end, since
    /// otherwise the optimization will be included in the reduce fusion block.
    pub fuser: ReduceFuser<R>,
    ops: Vec<OperationIr>,
    kind: ReduceBlockKind,
}

/// This type is responsible to fuse a single trace for all operations involved in this
/// optimization.
pub struct ReduceBroadcastedMainFuser {
    pub(crate) fuser: TraceOperationFuser,
    blocks: Vec<ReduceBlockKind>,
    settings_write: FuseSettings,
}

/// Where the fusion is at during the fusing process.
#[derive(Debug, Clone)]
pub enum ReduceBroadcastedStatus {
    /// The fusion is just starting, no reduce has been fused yet.
    Starting,
    /// The fusion is initialized with at least one reduce operation.
    ///
    /// # Notes
    ///
    /// The following reduce operation will need to be compatible with the previous reduction to
    /// fuse.
    Init { shape_id: Vec<usize>, axis: usize },
    /// No more operation can be fused.
    Closed,
}

/// The [ReduceBlockFuser] capacity to accept an [OperationIr].
#[derive(Clone, Copy, Debug)]
pub enum ReduceBlockFusionAnalysis {
    /// The operation can be fused, simply call [ReduceBlockFuser::fuse()].
    Accept,
    /// The operation can't be fused, the optimization should close.
    Refuse,
    /// The operation can be fused, but in another block.
    NewBlockRequired,
}

impl<R: Runtime> ReduceBlockFuser<R> {
    /// Creates a new block.
    pub fn new(fuser: ReduceFuser<R>) -> Self {
        Self {
            fuser: fuser.clone(),
            ops: Vec::new(),
            kind: ReduceBlockKind::Elemwise,
        }
    }

    /// Is an elemwise fuser.
    pub fn is_elemwise(&self) -> bool {
        matches!(self.kind, ReduceBlockKind::Elemwise)
    }
    /// Analyzes if fusion is possible within this block.
    pub fn analyze(
        &self,
        op: &OperationIr,
        status: &ReduceBroadcastedStatus,
        default_node: &ReduceFuser<R>,
    ) -> ReduceBlockFusionAnalysis {
        let mut fuser_try = self.fuser.clone();
        let before = fuser_try.len();
        fuser_try.fuse(op);
        let after = fuser_try.len();

        if after > before {
            return ReduceBlockFusionAnalysis::Accept;
        }

        let mut fuser_try = default_node.clone();

        let before = fuser_try.len();
        fuser_try.fuse(op);
        let after = fuser_try.len();

        if after > before {
            let info = fuser_try.reduce_info();

            return match (info, status) {
                (
                    ReduceFuserInfo::FusedReduce {
                        shape_input_id,
                        axis,
                    },
                    ReduceBroadcastedStatus::Init {
                        shape_id,
                        axis: axis_init,
                    },
                ) => {
                    if shape_id == &shape_input_id && axis_init == &axis {
                        ReduceBlockFusionAnalysis::NewBlockRequired
                    } else {
                        ReduceBlockFusionAnalysis::Refuse
                    }
                }
                (
                    ReduceFuserInfo::FusedElemwise { shape_id },
                    ReduceBroadcastedStatus::Init {
                        shape_id: shape_init,
                        ..
                    },
                ) => {
                    if &shape_id == shape_init {
                        ReduceBlockFusionAnalysis::NewBlockRequired
                    } else {
                        ReduceBlockFusionAnalysis::Refuse
                    }
                }
                _ => ReduceBlockFusionAnalysis::Refuse,
            };
        }

        ReduceBlockFusionAnalysis::Refuse
    }

    /// Fuses an operation within this block.
    ///
    /// # Warning
    ///
    /// It's important to call [Self::analyze_fusion()] before calling this function to make sure
    /// the current block accepts this operation.
    pub fn fuse(&mut self, op: &OperationIr) {
        self.fuser.fuse(op);
        self.ops.push(op.clone());

        // We update the kind if necessary.
        match (&self.fuser.reduce, &self.kind) {
            (Some(reduce), ReduceBlockKind::Elemwise) => {
                self.kind = ReduceBlockKind::Reduce {
                    // We just merged the reduce, since the last step the kind was Elemwise.
                    ops_index: self.ops.len() - 1,
                    reduce: reduce.clone(),
                };
            }
            _ => {}
        }
    }

    /// Computes the fuser properties.
    pub fn properties(&self) -> FuserProperties {
        let mut properties = self.fuser.properties();
        match &self.kind {
            // We can always run elementwise trace.
            ReduceBlockKind::Elemwise => properties.ready = true,
            ReduceBlockKind::Reduce { .. } => {}
        }
        properties
    }

    pub fn finish(
        &self,
        num_ops: &mut usize,
        main: &mut ReduceBroadcastedMainFuser,
    ) -> ReduceBlockOptimInfo<R> {
        main.register(self);

        match self.kind {
            ReduceBlockKind::Elemwise => {
                let len = self.fuser.fuser_read_fallback.len();
                let device = self.fuser.device.clone();
                *num_ops += len;
                let trace = self.fuser.fuser_read_fallback.finish();
                let client = R::client(&device);
                let elementwise = ElemwiseOptimization::new(trace, client, device, len);
                ReduceBlockOptimInfo::Elemwise(Arc::new(elementwise))
            }
            ReduceBlockKind::Reduce { .. } => {
                *num_ops += self.fuser.len();
                let optim = self.fuser.finish();
                let optim = match optim {
                    CubeOptimization::Reduce(optim) => optim.info,
                    _ => unreachable!(),
                };
                ReduceBlockOptimInfo::Reduce(optim)
            }
        }
    }
}

impl ReduceBroadcastedMainFuser {
    /// Creates a new fuser with the given settings.
    pub fn new(max_bindings: u32, bool_precision: FuseType) -> Self {
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
            // TODO: Fusion axis should be on the reduce_axis - 1.
            vectorization: VectorizationSetting::SmallerOrEqualThanPreviousBlock,
            ..Default::default()
        };
        Self {
            fuser,
            blocks: Vec::new(),
            settings_write,
        }
    }

    /// Registers a [ReduceBlockFuser] to build the trace.
    pub fn register<R: Runtime>(&mut self, block: &ReduceBlockFuser<R>) {
        match &block.kind {
            ReduceBlockKind::Elemwise => {
                // First we need to close the previous block.
                if !self.fuser.is_empty() {
                    self.fuser.next_block([], self.settings_write).unwrap();
                }
                // We register the new operations in a fresh block.
                for op in block.ops.iter() {
                    self.fuser.fuse(op);
                }
                self.blocks.push(ReduceBlockKind::Elemwise);
            }
            ReduceBlockKind::Reduce { ops_index, reduce } => {
                // First we need to close the previous block.
                if !self.fuser.is_empty() {
                    self.fuser.next_block([], self.settings_write).unwrap();
                }

                for op in &block.ops[0..*ops_index] {
                    self.fuser.fuse(op);
                }

                let [input] = self
                    .fuser
                    .next_block([&reduce.op.input], self.settings_write)
                    .unwrap();
                let output = self.fuser.output_unhandled(&reduce.op.out);
                let reduce = FusedReduce {
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
                    reduce,
                });
                let length = block.ops.len();
                for op in &block.ops[*ops_index..length] {
                    self.fuser.fuse(op);
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
enum ReduceBlockKind {
    Elemwise,
    Reduce {
        ops_index: usize,
        reduce: FusedReduce,
    },
}

impl<R: Runtime> Clone for ReduceBlockFuser<R> {
    fn clone(&self) -> Self {
        Self {
            fuser: self.fuser.clone(),
            ops: self.ops.clone(),
            kind: self.kind.clone(),
        }
    }
}
