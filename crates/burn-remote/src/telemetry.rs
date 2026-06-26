//! Best-effort server telemetry.
//!
//! Session workers emit [`TelemetryEvent`]s into a [`TelemetryProbe`]; a monitoring view holds a
//! [`TelemetrySubscription`] and renders them. The probe never blocks compute: events are built
//! only while a subscriber is attached and are dropped on lag, so a slow or idle viewer cannot
//! apply backpressure to a worker.

use std::collections::HashMap;
use std::sync::Arc;

use burn_backend::{DType, Shape};
use burn_ir::{GraphId, NumericOperationIr, OperationIr, TensorId};
use burn_std::id::StreamId;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::shared::{RequestId, SessionId};

/// A tensor as it appears in the dataflow graph: identity plus the shape and dtype it was
/// produced with.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorRef {
    pub id: TensorId,
    pub shape: Shape,
    pub dtype: DType,
}

/// One operation inside a cached graph, in relative form (positional tensor ids).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphOp {
    pub kind: OpClass,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorRef>,
}

impl GraphOp {
    pub(crate) fn summarize(op: &OperationIr) -> Self {
        Self {
            kind: classify(op),
            inputs: op.inputs().map(|tensor| tensor.id).collect(),
            outputs: op
                .outputs()
                .map(|tensor| TensorRef {
                    id: tensor.id,
                    shape: tensor.shape.clone(),
                    dtype: tensor.dtype,
                })
                .collect(),
        }
    }
}

/// Coarse operation category, used to colour and aggregate the op stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpClass {
    Matmul,
    Conv,
    Linear,
    Activation,
    Reduction,
    Elementwise,
    Compare,
    Cast,
    Index,
    Reshape,
    Init,
    Random,
    Drop,
    Custom,
    Distributed,
    Other,
}

impl OpClass {
    /// Every variant, in a stable order for layout and iteration.
    pub const ALL: [OpClass; 16] = [
        OpClass::Matmul,
        OpClass::Conv,
        OpClass::Linear,
        OpClass::Activation,
        OpClass::Reduction,
        OpClass::Elementwise,
        OpClass::Compare,
        OpClass::Cast,
        OpClass::Index,
        OpClass::Reshape,
        OpClass::Init,
        OpClass::Random,
        OpClass::Drop,
        OpClass::Custom,
        OpClass::Distributed,
        OpClass::Other,
    ];

    /// Stable lowercase label for display and aggregation.
    pub fn label(self) -> &'static str {
        match self {
            OpClass::Matmul => "matmul",
            OpClass::Conv => "conv",
            OpClass::Linear => "linear",
            OpClass::Activation => "activation",
            OpClass::Reduction => "reduction",
            OpClass::Elementwise => "elementwise",
            OpClass::Compare => "compare",
            OpClass::Cast => "cast",
            OpClass::Index => "index",
            OpClass::Reshape => "reshape",
            OpClass::Init => "init",
            OpClass::Random => "random",
            OpClass::Drop => "drop",
            OpClass::Custom => "custom",
            OpClass::Distributed => "distributed",
            OpClass::Other => "other",
        }
    }
}

/// Classify an [`OperationIr`] into a coarse [`OpClass`].
pub fn classify(op: &OperationIr) -> OpClass {
    use OperationIr as O;
    match op {
        O::NumericFloat(_, n) | O::NumericInt(_, n) => classify_numeric(n),
        O::Float(_, f) => classify_float(f),
        O::Module(m) => classify_module(m),
        O::Activation(_) => OpClass::Activation,
        O::BaseFloat(b) | O::BaseInt(b) | O::BaseBool(b) => classify_base(b),
        O::Init(_) => OpClass::Init,
        O::Custom(_) => OpClass::Custom,
        O::Drop(_) => OpClass::Drop,
        O::Distributed(_) => OpClass::Distributed,
        O::Bool(_) | O::Int(_) => OpClass::Elementwise,
    }
}

fn classify_numeric(op: &NumericOperationIr) -> OpClass {
    use NumericOperationIr as N;
    match op {
        N::Mean(_)
        | N::MeanDim(_)
        | N::Sum(_)
        | N::SumDim(_)
        | N::Prod(_)
        | N::ProdDim(_)
        | N::Max(_)
        | N::MaxDim(_)
        | N::MaxDimWithIndices(_)
        | N::Min(_)
        | N::MinDim(_)
        | N::MinDimWithIndices(_)
        | N::MaxAbs(_)
        | N::MaxAbsDim(_)
        | N::ArgMax(_)
        | N::ArgMin(_)
        | N::ArgTopK(_)
        | N::TopK(_)
        | N::CumSum(_)
        | N::CumProd(_)
        | N::CumMin(_)
        | N::CumMax(_)
        | N::Sort(_)
        | N::SortWithIndices(_)
        | N::ArgSort(_) => OpClass::Reduction,
        N::Greater(_)
        | N::GreaterElem(_)
        | N::GreaterEqual(_)
        | N::GreaterEqualElem(_)
        | N::Lower(_)
        | N::LowerElem(_)
        | N::LowerEqual(_)
        | N::LowerEqualElem(_) => OpClass::Compare,
        N::Full(_) | N::IntRandom(_) => OpClass::Init,
        _ => OpClass::Elementwise,
    }
}

fn classify_float(op: &burn_ir::FloatOperationIr) -> OpClass {
    use burn_ir::FloatOperationIr as F;
    match op {
        F::Matmul(_) => OpClass::Matmul,
        F::Random(_) => OpClass::Random,
        F::IntoInt(_) | F::Quantize(_) | F::Dequantize(_) => OpClass::Cast,
        _ => OpClass::Elementwise,
    }
}

fn classify_module(op: &burn_ir::ModuleOperationIr) -> OpClass {
    let name = format!("{op:?}");
    if name.starts_with("Linear") || name.starts_with("Embedding") {
        OpClass::Linear
    } else if name.starts_with("Conv") || name.starts_with("DeformableConv") {
        OpClass::Conv
    } else {
        OpClass::Other
    }
}

fn classify_base(op: &burn_ir::BaseOperationIr) -> OpClass {
    use burn_ir::BaseOperationIr as B;
    match op {
        B::Reshape(_)
        | B::SwapDims(_)
        | B::Permute(_)
        | B::Flip(_)
        | B::Expand(_)
        | B::RepeatDim(_)
        | B::Cat(_)
        | B::Unfold(_) => OpClass::Reshape,
        B::Cast(_) => OpClass::Cast,
        B::Empty(_) | B::Ones(_) | B::Zeros(_) => OpClass::Init,
        B::Equal(_)
        | B::EqualElem(_)
        | B::NotEqual(_)
        | B::NotEqualElem(_)
        | B::All(_)
        | B::Any(_)
        | B::AllDim(_)
        | B::AnyDim(_) => OpClass::Compare,
        _ => OpClass::Index,
    }
}

/// Whether a transfer crosses hosts or stays on this machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferScope {
    Local,
    Remote,
}

/// Lifecycle point of a tensor transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferPhase {
    Started,
    Completed,
    Failed,
}

/// A single observation from a session worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TelemetryEvent {
    SessionOpened {
        session: SessionId,
        device: u32,
    },
    SessionClosed {
        session: SessionId,
    },
    Op {
        session: SessionId,
        stream: StreamId,
        kind: OpClass,
        inputs: Vec<TensorId>,
        outputs: Vec<TensorRef>,
    },
    TensorDropped {
        session: SessionId,
        tensor: TensorId,
    },
    Transfer {
        session: SessionId,
        peer: Option<String>,
        scope: TransferScope,
        phase: TransferPhase,
    },
    Read {
        session: SessionId,
        request: RequestId,
    },
    Sync {
        session: SessionId,
        request: RequestId,
    },
    GraphRegistered {
        session: SessionId,
        graph: GraphId,
        ops: Vec<GraphOp>,
        bytes: usize,
    },
    GraphExecuted {
        session: SessionId,
        graph: GraphId,
        stream: StreamId,
        bindings_bytes: usize,
    },
}

impl TelemetryEvent {
    pub(crate) fn unfused_op(session: SessionId, stream: StreamId, op: &OperationIr) -> Self {
        match op {
            OperationIr::Drop(tensor) => TelemetryEvent::TensorDropped {
                session,
                tensor: tensor.id,
            },
            _ => {
                let GraphOp {
                    kind,
                    inputs,
                    outputs,
                } = GraphOp::summarize(op);
                TelemetryEvent::Op {
                    session,
                    stream,
                    kind,
                    inputs,
                    outputs,
                }
            }
        }
    }

    pub(crate) fn graph_registered(
        session: SessionId,
        graph: GraphId,
        ops: &[OperationIr],
        bytes: usize,
    ) -> Self {
        TelemetryEvent::GraphRegistered {
            session,
            graph,
            ops: ops.iter().map(GraphOp::summarize).collect(),
            bytes,
        }
    }

    pub(crate) fn graph_executed(
        session: SessionId,
        graph: GraphId,
        stream: StreamId,
        bindings_bytes: usize,
    ) -> Self {
        TelemetryEvent::GraphExecuted {
            session,
            graph,
            stream,
            bindings_bytes,
        }
    }
}

pub(crate) const CHANNEL_CAPACITY: usize = 4096;

/// Cloneable handle a worker emits into. Inert until a [`TelemetrySubscription`] is attached.
#[derive(Clone)]
pub struct TelemetryProbe {
    tx: Option<broadcast::Sender<Arc<TelemetryEvent>>>,
}

impl TelemetryProbe {
    /// A probe that drops everything. Worker emit points compile to a cheap no-op against it.
    pub fn disabled() -> Self {
        Self { tx: None }
    }

    /// Create an active probe with no initial subscription. Subscribers attach later with
    /// [`subscribe`](Self::subscribe); the probe stays inert until at least one is listening.
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity);
        Self { tx: Some(tx) }
    }

    /// Create an active probe and its first subscription. `capacity` bounds the per-subscriber
    /// backlog; older events are dropped once a subscriber falls that far behind.
    pub fn channel(capacity: usize) -> (Self, TelemetrySubscription) {
        let (tx, rx) = broadcast::channel(capacity);
        (Self { tx: Some(tx) }, TelemetrySubscription { rx })
    }

    /// Attach another subscription, or `None` if this probe is disabled.
    pub fn subscribe(&self) -> Option<TelemetrySubscription> {
        self.tx
            .as_ref()
            .map(|tx| TelemetrySubscription { rx: tx.subscribe() })
    }

    /// Whether building and emitting an event is worthwhile right now.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.tx.as_ref().is_some_and(|tx| tx.receiver_count() > 0)
    }

    /// Emit an event, building it only when a subscriber is listening.
    #[inline]
    pub fn emit(&self, event: impl FnOnce() -> TelemetryEvent) {
        if let Some(tx) = &self.tx
            && tx.receiver_count() > 0
        {
            let _ = tx.send(Arc::new(event()));
        }
    }
}

/// Receiving end of a [`TelemetryProbe`].
pub struct TelemetrySubscription {
    rx: broadcast::Receiver<Arc<TelemetryEvent>>,
}

/// Outcome of a non-blocking drain.
pub enum DrainStatus {
    /// The channel is still open; `lagged` counts events dropped before this drain.
    Open { lagged: u64 },
    /// Every sender has been dropped.
    Closed,
}

impl TelemetrySubscription {
    /// Drain all currently buffered events into `sink` without blocking, reporting whether the
    /// channel is still open and how many events were dropped to lag. Intended to be called once
    /// per UI frame.
    pub fn drain_into(&mut self, sink: &mut Vec<Arc<TelemetryEvent>>) -> DrainStatus {
        use broadcast::error::TryRecvError;
        let mut lagged = 0;
        loop {
            match self.rx.try_recv() {
                Ok(event) => sink.push(event),
                Err(TryRecvError::Empty) => return DrainStatus::Open { lagged },
                Err(TryRecvError::Lagged(n)) => lagged += n,
                Err(TryRecvError::Closed) => return DrainStatus::Closed,
            }
        }
    }

    /// Await the next event, skipping lag gaps. Returns `None` once every sender is dropped.
    pub async fn recv(&mut self) -> Option<Arc<TelemetryEvent>> {
        use broadcast::error::RecvError;
        loop {
            match self.rx.recv().await {
                Ok(event) => return Some(event),
                Err(RecvError::Lagged(_)) => continue,
                Err(RecvError::Closed) => return None,
            }
        }
    }
}

struct GraphCost {
    bytes: u64,
    ops: u64,
}

/// A running fold of the op-graph caching economics over a telemetry stream.
#[derive(Default)]
pub struct TrafficAggregator {
    graphs: HashMap<GraphId, GraphCost>,
    baseline: u64,
    actual: u64,
    fused_ops: u64,
    unfused_ops: u64,
    unfused_by_kind: HashMap<OpClass, u64>,
}

impl TrafficAggregator {
    pub fn apply(&mut self, event: &TelemetryEvent) {
        match event {
            TelemetryEvent::GraphRegistered {
                graph, ops, bytes, ..
            } => {
                self.actual += *bytes as u64;
                self.graphs.insert(
                    *graph,
                    GraphCost {
                        bytes: *bytes as u64,
                        ops: ops.len() as u64,
                    },
                );
            }
            TelemetryEvent::GraphExecuted {
                graph,
                bindings_bytes,
                ..
            } => {
                if let Some(cost) = self.graphs.get(graph) {
                    self.baseline += cost.bytes;
                    self.fused_ops += cost.ops;
                }
                self.actual += *bindings_bytes as u64;
            }
            TelemetryEvent::Op { kind, .. } => {
                self.unfused_ops += 1;
                *self.unfused_by_kind.entry(*kind).or_default() += 1;
            }
            TelemetryEvent::TensorDropped { .. } => {
                self.unfused_ops += 1;
                *self.unfused_by_kind.entry(OpClass::Drop).or_default() += 1;
            }
            _ => {}
        }
    }

    pub fn snapshot(&self) -> FusionSnapshot {
        FusionSnapshot {
            fused_ops: self.fused_ops,
            unfused_ops: self.unfused_ops,
            baseline: self.baseline,
            actual: self.actual,
        }
    }

    /// Unfused-op tally by class, highest first.
    pub fn unfused_by_kind(&self) -> Vec<(OpClass, u64)> {
        let mut entries: Vec<_> = self
            .unfused_by_kind
            .iter()
            .map(|(kind, count)| (*kind, *count))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.label().cmp(b.0.label())));
        entries
    }
}

pub struct FusionSnapshot {
    pub fused_ops: u64,
    pub unfused_ops: u64,
    pub baseline: u64,
    pub actual: u64,
}

impl FusionSnapshot {
    /// Baseline a non-fusion peer would have moved, minus what fusion actually moved.
    pub fn saved(&self) -> u64 {
        self.baseline.saturating_sub(self.actual)
    }

    pub fn total_ops(&self) -> u64 {
        self.fused_ops + self.unfused_ops
    }
}

pub(crate) fn serialized_len<T: Serialize>(value: &T) -> usize {
    rmp_serde::to_vec(value)
        .map(|bytes| bytes.len())
        .unwrap_or(0)
}
