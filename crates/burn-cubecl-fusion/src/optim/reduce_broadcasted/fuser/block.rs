use crate::optim::{
    CubeOptimization,
    elemwise::ElemwiseOptimization,
    reduce::{FusedReduce, ReduceFuser, ReduceFuserInfo},
    reduce_broadcasted::{ReduceBlockOptimInfo, fuser::full::ReduceBroadcastedFullFuser},
};
use burn_fusion::{FuserProperties, OperationFuser};
use burn_ir::OperationIr;
use cubecl::Runtime;
use std::sync::Arc;

/// Responsible for fusing a single reduce block or elementwise block.
///
/// When the block kind is reduce, it supports fuse-on-read and fuse-on-write fusion.
/// Broadcasting isn't supported; another block should handle it instead.
pub struct ReduceBlockFuser<R: Runtime> {
    /// We use [ReduceFuser] for both elementwise and reduce blocks, keeping only the
    /// fuse-on-read trace if the block is tagged as elementwise.
    ///
    /// # Notes
    ///
    /// A single elementwise block can only exist at the end of a full [ReduceBlockFuser],
    /// otherwise the optimization will be included in the reduce fusion block.
    pub fuser: ReduceFuser<R>,
    pub(crate) ops: Vec<OperationIr>,
    pub(crate) kind: ReduceBlockKind,
}

/// The current state of the fusion process.
#[derive(Debug, Clone)]
pub enum ReduceBroadcastedStatus {
    /// Fusion is starting; no reduction has been fused yet.
    Starting,
    /// Fusion is initialized with at least one reduce operation.
    ///
    /// # Notes
    ///
    /// Subsequent reduce operations must be compatible with the previous reduction to fuse.
    Init { shape_id: Vec<usize>, axis: usize },
    /// No more operations can be fused.
    Closed,
    /// Invalid axis.
    Abort,
}

/// The [ReduceBlockFuser] capacity to accept an [OperationIr].
#[derive(Clone, Copy, Debug)]
pub enum ReduceBlockFusionAnalysis {
    /// The operation can be fused; call [ReduceBlockFuser::fuse()].
    Accept,
    /// The operation cannot be fused; the optimization should close.
    Refuse,
    /// The operation can be fused, but requires a new block.
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

    /// Returns true if this is an elementwise fuser.
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

        // Can't create a new block if the previous one was not a reduction.
        if self.fuser.reduce.is_none() {
            return ReduceBlockFusionAnalysis::Refuse;
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
    /// Ensure [Self::analyze()] is called before this function to confirm the operation is accepted.
    pub fn fuse(&mut self, op: &OperationIr) {
        self.fuser.fuse(op);
        self.ops.push(op.clone());

        // Update the kind if a reduction is introduced to an elementwise block.
        if let (Some(reduce), ReduceBlockKind::Elemwise) = (&self.fuser.reduce, &self.kind) {
            self.kind = ReduceBlockKind::Reduce {
                ops_index: self.ops.len() - 1,
                reduce: Box::new(reduce.clone()),
            };
        }
    }

    /// Computes the fuser properties.
    pub fn properties(&self) -> FuserProperties {
        let mut properties = self.fuser.properties();
        if let ReduceBlockKind::Elemwise = &self.kind {
            // Elementwise traces are always ready to run.
            properties.ready = true;
        }
        properties
    }

    pub fn finish(
        &mut self,
        num_ops: &mut usize,
        full: &mut ReduceBroadcastedFullFuser,
    ) -> ReduceBlockOptimInfo<R> {
        full.register(self);

        match &self.kind {
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
                let info = match optim {
                    CubeOptimization::Reduce(optim) => optim.info,
                    _ => unreachable!("Expected Reduce optimization"),
                };
                ReduceBlockOptimInfo::Reduce(info)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum ReduceBlockKind {
    Elemwise,
    Reduce {
        ops_index: usize,
        reduce: Box<FusedReduce>,
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
