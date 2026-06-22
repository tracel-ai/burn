//! Network-traffic savings metric for the remote backend's op-graph caching (fusion).
//!
//! Each replay of a cached graph carries only its per-invocation bindings instead of the full op
//! stream a non-fusion peer would re-send every time. A [`TrafficMetrics`] accumulator tracks how
//! many bytes that saves and logs it via [`log_remote`], when remote logging is enabled (`[remote]`
//! section in `burn.toml`, or `BURN_REMOTE_LOG`).
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

/// Accumulates the network-traffic savings of op-graph caching for one endpoint.
///
/// Not shared and not global: each instance is owned by the single thread that records into it (the
/// client's device-runner thread, or a server session's worker thread), so it needs no locking or
/// atomics.
pub(crate) struct TrafficMetrics {
    /// Whether this accumulator measures the client or the server end, for log labelling.
    side: MetricSide,
    /// Serialized size (bytes) of each registered graph, to value its replays.
    graph_sizes: HashMap<GraphId, usize>,
    /// Bytes a non-fusion peer would have streamed for the covered work (the serialized graph,
    /// counted once per replay).
    baseline: u64,
    /// Bytes actually moved: each graph once, plus the bindings of every replay.
    actual: u64,
    /// Highest whole-MiB savings already reported, so `Basic` logging emits at most one line per
    /// additional mebibyte saved instead of one per replay.
    logged_mib: u64,
}

impl TrafficMetrics {
    /// A fresh accumulator for the given side of the connection.
    pub(crate) fn new(side: MetricSide) -> Self {
        Self {
            side,
            graph_sizes: HashMap::new(),
            baseline: 0,
            actual: 0,
            logged_mib: 0,
        }
    }

    /// Record that `graph` was registered once under `id` (the one-time cost of caching it).
    pub(crate) fn record_registration(&mut self, id: GraphId, graph: &[OperationIr]) {
        if level() == RemoteLogLevel::Disabled {
            return;
        }

        let size = serialized_len(&graph);
        self.graph_sizes.insert(id, size);
        self.actual += size as u64;

        let side = self.side;
        log_remote(RemoteLogLevel::Full, || {
            format!("[remote {side}] registered graph {id:?}: {size} bytes (sent once)")
        });
    }

    /// Record a replay of graph `id`: only `bindings` moved instead of the full graph.
    pub(crate) fn record_execution(&mut self, id: GraphId, bindings: &GraphBindings) {
        if level() == RemoteLogLevel::Disabled {
            return;
        }

        let bindings_size = serialized_len(bindings);
        let graph_size = self.graph_sizes.get(&id).copied().unwrap_or(0);

        self.actual += bindings_size as u64;
        self.baseline += graph_size as u64;

        let saved = self.baseline.saturating_sub(self.actual);
        let pct = percent_saved(saved, self.baseline);
        let side = self.side;

        if level() >= RemoteLogLevel::Full {
            log_remote(RemoteLogLevel::Full, || {
                format!(
                    "[remote {side}] replayed graph {id:?}: {bindings_size} bytes instead of \
                     ~{graph_size}; cumulative saved {saved} bytes ({pct:.1}% of baseline)"
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
                         {pct:.1}% of baseline) of network traffic"
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

/// Savings as a percentage of the baseline — the fraction of would-be traffic that caching avoided.
/// Zero while the baseline is still zero (nothing has been replayed yet), so the first log line
/// can't divide by zero.
fn percent_saved(saved: u64, baseline: u64) -> f64 {
    if baseline == 0 {
        0.0
    } else {
        saved as f64 / baseline as f64 * 100.0
    }
}
