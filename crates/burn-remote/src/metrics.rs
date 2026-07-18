//! Logs op-graph caching (fusion) savings as `[remote ...]` lines, gated behind the runtime log
//! level (`[remote]` in `burn.toml`, or `BURN_REMOTE_LOG`). A best-effort [`TelemetryProbe`]
//! subscriber; client and server each run their own, distinguished by a [`MetricSide`] label.
//!
//! [`TelemetryProbe`]: crate::telemetry::TelemetryProbe

use core::fmt;

use burn_ir::GraphId;
use burn_std::config::config;
use burn_std::config::log_remote;
use burn_std::config::remote::RemoteLogLevel;

use crate::telemetry::{
    OpClass, TelemetryEvent, TelemetryProbe, TelemetrySubscription, TrafficAggregator,
};

const MIB: u64 = 1024 * 1024;

/// Which side of the connection a logger reports, used to label its log lines.
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

/// A best-effort telemetry subscriber that logs op-graph caching savings.
pub(crate) struct TelemetryLogger {
    side: MetricSide,
    aggregator: TrafficAggregator,
    /// Highest whole-MiB savings already reported, so `Basic` logging emits at most one line per
    /// additional mebibyte saved instead of one per replay.
    logged_mib: u64,
}

impl TelemetryLogger {
    pub(crate) fn new(side: MetricSide) -> Self {
        Self {
            side,
            aggregator: TrafficAggregator::default(),
            logged_mib: 0,
        }
    }

    /// Whether remote logging is configured.
    pub(crate) fn enabled() -> bool {
        level() != RemoteLogLevel::Disabled
    }

    /// Drain and log until the probe's senders are dropped.
    pub(crate) async fn run(mut self, mut events: TelemetrySubscription) {
        while let Some(event) = events.recv().await {
            self.aggregator.apply(&event);
            match event.as_ref() {
                TelemetryEvent::GraphRegistered {
                    graph, ops, bytes, ..
                } => self.log_registration(*graph, ops.len(), *bytes),
                TelemetryEvent::GraphExecuted { .. } => self.log_progress(),
                _ => {}
            }
        }
    }

    fn log_registration(&self, graph: GraphId, ops: usize, bytes: usize) {
        let side = self.side;
        log_remote(RemoteLogLevel::Full, || {
            format!(
                "[remote {side}] registered graph {graph:?}: {ops} ops, {bytes} bytes (sent once)"
            )
        });
    }

    fn log_progress(&mut self) {
        let snapshot = self.aggregator.snapshot();
        let saved = snapshot.saved();
        let saved_pct = percentage(saved, snapshot.baseline);
        let side = self.side;

        if level() >= RemoteLogLevel::Full {
            let fused_pct = percentage(snapshot.fused_ops, snapshot.total_ops());
            let breakdown = format_unfused_by_kind(&self.aggregator.unfused_by_kind());
            log_remote(RemoteLogLevel::Full, || {
                format!(
                    "[remote {side}] op-graph caching saved {saved} bytes ({saved_pct:.1}% of \
                     baseline); fused {fused_pct:.1}% of {} ops total; unfused by kind: [{breakdown}]",
                    snapshot.total_ops()
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

/// The logger's drain loop for `probe`, or `None` when remote logging is off. Spawn it detached.
pub(crate) fn logger_task(
    probe: &TelemetryProbe,
    side: MetricSide,
) -> Option<impl core::future::Future<Output = ()> + Send + 'static> {
    if !TelemetryLogger::enabled() {
        return None;
    }
    let subscription = probe.subscribe()?;
    Some(TelemetryLogger::new(side).run(subscription))
}

fn level() -> RemoteLogLevel {
    config().remote().logger.level
}

fn format_unfused_by_kind(entries: &[(OpClass, u64)]) -> String {
    entries
        .iter()
        .map(|(kind, count)| format!("{}={count}", kind.label()))
        .collect::<Vec<_>>()
        .join(", ")
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
