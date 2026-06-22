//! Network-traffic savings metric for the remote backend's op-graph caching (fusion).
//!
//! Each replay of a cached graph carries only its per-invocation bindings instead of the full op
//! stream a non-fusion peer would re-send every time. A [`TrafficMetrics`] accumulator tracks how
//! many bytes that saves — and, at `full` log level, how many ops run fused (via cached graphs) vs
//! unfused (streamed one-by-one) — and logs it via [`log_remote`], when remote logging is enabled
//! (`[remote]` section in `burn.toml`, or `BURN_REMOTE_LOG`).
//!
//! The accumulator is held per endpoint rather than in process globals: the client keeps one in its
//! per-device service (`RemoteService`) and the server keeps one per session (`SessionHandler`), so
//! each side measures its own traffic. Both sides see the same byte counts — they serialize and
//! deserialize the same IR — so one type serves both, distinguished only by a [`MetricSide`] label
//! in the log lines.
//!
//! All work is gated behind the runtime log level, so a disabled metric costs only one config read
//! and adds nothing to the hot path.

use core::fmt;
use std::collections::HashMap;

use burn_ir::{GraphBindings, GraphId, OperationIr};
use burn_std::config::config;
use burn_std::config::log_remote;
use burn_std::config::remote::RemoteLogLevel;

const MIB: u64 = 1024 * 1024;

/// Which side of the connection an accumulator measures, used only to label its log lines.
#[derive(Clone, Copy, Debug)]
pub(crate) enum MetricSide {
    /// The client: bytes it would have sent vs. bytes it actually sent.
    Client,
    /// The server: bytes it would have received vs. bytes it actually received.
    Server,
}

impl fmt::Display for MetricSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricSide::Client => f.write_str("client"),
            MetricSide::Server => f.write_str("server"),
        }
    }
}

/// What a registered graph is worth, to value its replays.
struct GraphInfo {
    /// Serialized size (bytes), re-sent on every replay by a non-fusion peer.
    bytes: usize,
    /// Number of operations the graph fuses into a single cached unit.
    ops: usize,
}

/// Accumulates the network-traffic savings of op-graph caching for one endpoint.
///
/// Not shared and not global: each instance is owned by the single thread that records into it (the
/// client's device-runner thread, or a server session's worker thread), so it needs no locking or
/// atomics.
pub(crate) struct TrafficMetrics {
    /// Whether this accumulator measures the client or the server end, for log labelling.
    side: MetricSide,
    /// Per-registered-graph size and op count, to value each replay.
    graphs: HashMap<GraphId, GraphInfo>,
    /// Bytes a non-fusion peer would have streamed for the covered work (the serialized graph,
    /// counted once per replay).
    baseline: u64,
    /// Bytes actually moved: each graph once, plus the bindings of every replay.
    actual: u64,
    /// Highest whole-MiB savings already reported, so `Basic` logging emits at most one line per
    /// additional mebibyte saved instead of one per replay.
    logged_mib: u64,
    /// Ops executed via cached graphs, counted on every graph execution (including replays).
    fused_ops: u64,
    /// Ops executed one-by-one, outside any cached graph.
    unfused_ops: u64,
    /// Unfused ops seen since the last graph execution — the run of unfused ops immediately
    /// preceding the next graph. Reset each time a graph executes.
    unfused_before_graph: u64,
}

impl TrafficMetrics {
    /// A fresh accumulator for the given side of the connection.
    pub(crate) fn new(side: MetricSide) -> Self {
        Self {
            side,
            graphs: HashMap::new(),
            baseline: 0,
            actual: 0,
            logged_mib: 0,
            fused_ops: 0,
            unfused_ops: 0,
            unfused_before_graph: 0,
        }
    }

    /// Record one operation executed outside any cached graph (an unfused op).
    pub(crate) fn record_unfused_op(&mut self) {
        if level() == RemoteLogLevel::Disabled {
            return;
        }
        self.unfused_ops += 1;
        self.unfused_before_graph += 1;
    }

    /// Record that `graph` was registered once under `id` (the one-time cost of caching it).
    pub(crate) fn record_registration(&mut self, id: GraphId, graph: &[OperationIr]) {
        if level() == RemoteLogLevel::Disabled {
            return;
        }

        let bytes = serialized_len(&graph);
        let ops = graph.len();
        self.graphs.insert(id, GraphInfo { bytes, ops });
        self.actual += bytes as u64;

        let side = self.side;
        log_remote(RemoteLogLevel::Full, || {
            format!("[remote {side}] registered graph {id:?}: {ops} ops, {bytes} bytes (sent once)")
        });
    }

    /// Record a replay of graph `id`: only `bindings` moved instead of the full graph.
    pub(crate) fn record_execution(&mut self, id: GraphId, bindings: &GraphBindings) {
        if level() == RemoteLogLevel::Disabled {
            return;
        }

        let bindings_size = serialized_len(bindings);
        let (graph_bytes, graph_ops) = self
            .graphs
            .get(&id)
            .map(|g| (g.bytes, g.ops))
            .unwrap_or((0, 0));

        self.actual += bindings_size as u64;
        self.baseline += graph_bytes as u64;
        self.fused_ops += graph_ops as u64;

        let saved = self.baseline.saturating_sub(self.actual);
        let saved_pct = percentage(saved, self.baseline);

        // The run of unfused ops leading up to this graph; reset for the next run.
        let unfused_before = self.unfused_before_graph;
        self.unfused_before_graph = 0;

        let side = self.side;

        if level() >= RemoteLogLevel::Full {
            let total_ops = self.fused_ops + self.unfused_ops;
            let fused_pct = percentage(self.fused_ops, total_ops);
            log_remote(RemoteLogLevel::Full, || {
                format!(
                    "[remote {side}] replayed graph {id:?}: {bindings_size} bytes instead of \
                     ~{graph_bytes}; cumulative saved {saved} bytes ({saved_pct:.1}% of baseline); \
                     graph {graph_ops} ops, {unfused_before} unfused before it; \
                     fused {fused_pct:.1}% of {total_ops} ops total"
                )
            });
        } else {
            // Basic: self-throttle to one summary per additional mebibyte saved.
            let mib = saved / MIB;
            if mib > self.logged_mib {
                self.logged_mib = mib;
                log_remote(RemoteLogLevel::Basic, || {
                    format!(
                        "[remote {side}] op-graph caching has saved ~{mib} MiB ({saved} bytes, \
                         {saved_pct:.1}% of baseline) of network traffic"
                    )
                });
            }
        }
    }
}

fn level() -> RemoteLogLevel {
    config().remote().logger.level
}

fn serialized_len<T: serde::Serialize>(value: &T) -> usize {
    rmp_serde::to_vec(value)
        .map(|bytes| bytes.len())
        .unwrap_or(0)
}

/// `part` as a percentage of `whole`. Zero when `whole` is 0 (nothing measured yet), so the first
/// log line can't divide by zero.
fn percentage(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64 * 100.0
    }
}
