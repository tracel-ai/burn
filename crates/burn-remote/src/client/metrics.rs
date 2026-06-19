//! Network-traffic savings metric for the remote backend's op-graph caching (fusion).
//!
//! Each replay of a cached optimization sends only its per-invocation bindings instead of the full
//! op stream a non-fusion client would re-send every time. This module tracks, when remote logging
//! is enabled (`[remote]` section in `burn.toml`, or `BURN_REMOTE_LOG`), how many bytes that saves
//! and logs it via [`log_remote`].
//!
//! All work is gated behind the runtime log level, so a disabled metric costs only one atomic-free
//! config read and adds nothing to the hot path.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use burn_ir::{GraphBindings, GraphId, OperationIr};
use burn_std::config::config;
use burn_std::config::log_remote;
use burn_std::config::remote::RemoteLogLevel;

const MIB: u64 = 1024 * 1024;

/// Serialized size (bytes) of each registered graph, to value its replays.
static GRAPH_SIZES: Mutex<Option<HashMap<GraphId, usize>>> = Mutex::new(None);
/// Bytes a non-fusion client would have streamed for the optimization-covered work (approximated
/// by the serialized graph re-sent on every replay).
static BASELINE: AtomicU64 = AtomicU64::new(0);
/// Bytes actually sent: each graph once, plus the bindings of every replay.
static ACTUAL: AtomicU64 = AtomicU64::new(0);
/// Highest whole-MiB savings already reported, so `Basic` logging emits at most one line per
/// additional mebibyte saved instead of one per replay.
static LOGGED_MIB: AtomicU64 = AtomicU64::new(0);

fn level() -> RemoteLogLevel {
    config().remote().logger.level
}

fn serialized_len<T: serde::Serialize>(value: &T) -> usize {
    rmp_serde::to_vec(value)
        .map(|bytes| bytes.len())
        .unwrap_or(0)
}

/// Savings as a percentage of the baseline — the fraction of would-be traffic that caching
/// avoided. Zero while the baseline is still zero (nothing has been replayed yet), so the first
/// log line can't divide by zero.
fn percent_saved(saved: u64, baseline: u64) -> f64 {
    if baseline == 0 {
        0.0
    } else {
        saved as f64 / baseline as f64 * 100.0
    }
}

/// Record that `graph` was registered once under `id` (the one-time cost of caching it).
pub(crate) fn record_registration(id: GraphId, graph: &[OperationIr]) {
    if level() == RemoteLogLevel::Disabled {
        return;
    }

    let size = serialized_len(&graph);
    GRAPH_SIZES
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(id, size);
    ACTUAL.fetch_add(size as u64, Ordering::Relaxed);

    log_remote(RemoteLogLevel::Full, || {
        format!("[remote] registered graph {id:?}: {size} bytes (sent once)")
    });
}

/// Record a replay of graph `id`: we sent `bindings` instead of re-streaming the graph.
pub(crate) fn record_execution(id: GraphId, bindings: &GraphBindings) {
    if level() == RemoteLogLevel::Disabled {
        return;
    }

    let bindings_size = serialized_len(bindings);
    let graph_size = GRAPH_SIZES
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|sizes| sizes.get(&id).copied())
        .unwrap_or(0);

    ACTUAL.fetch_add(bindings_size as u64, Ordering::Relaxed);
    BASELINE.fetch_add(graph_size as u64, Ordering::Relaxed);

    let baseline = BASELINE.load(Ordering::Relaxed);
    let saved = baseline.saturating_sub(ACTUAL.load(Ordering::Relaxed));
    let pct = percent_saved(saved, baseline);

    if level() >= RemoteLogLevel::Full {
        log_remote(RemoteLogLevel::Full, || {
            format!(
                "[remote] replayed graph {id:?}: sent {bindings_size} bytes instead of \
                 ~{graph_size}; cumulative saved {saved} bytes ({pct:.1}% of baseline)"
            )
        });
    } else {
        // Basic: self-throttle to one summary per additional mebibyte saved.
        let mib = saved / MIB;
        let previous = LOGGED_MIB.fetch_max(mib, Ordering::Relaxed);
        if mib > previous {
            log_remote(RemoteLogLevel::Basic, || {
                format!(
                    "[remote] op-graph caching has saved ~{mib} MiB ({saved} bytes, \
                     {pct:.1}% of baseline) of network traffic"
                )
            });
        }
    }
}
