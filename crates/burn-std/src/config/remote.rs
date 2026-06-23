use cubecl_common::config::logger::{LogLevel, LoggerConfig};

/// Configuration for the remote backend.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RemoteConfig {
    /// Logger configuration for remote-backend logs (e.g. the bytes-saved metric of the
    /// fusion / op-graph caching feature).
    #[serde(default)]
    pub logger: LoggerConfig<RemoteLogLevel>,

    /// Flush the outgoing task buffer once this many tasks have accumulated.
    ///
    /// Wire-level batching only — every task keeps its own stream/request id, so the server sees
    /// the same per-task semantics. Larger values batch more aggressively (fewer, bigger frames);
    /// smaller values cut latency for chains of fire-and-forget submits.
    #[serde(default = "default_flush_threshold")]
    pub flush_threshold: usize,

    /// Flush once this many bytes of buffered tensor data accumulate, independent of
    /// [`flush_threshold`](Self::flush_threshold).
    ///
    /// Bounds how much tensor data sits unsent so large uploads go out promptly while small ops keep
    /// batching. Larger batches fewer/bigger frames; smaller cuts latency for data-heavy streams.
    #[serde(default = "default_flush_bytes_threshold")]
    pub flush_bytes_threshold: usize,
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self {
            logger: LoggerConfig::default(),
            flush_threshold: default_flush_threshold(),
            flush_bytes_threshold: default_flush_bytes_threshold(),
        }
    }
}

fn default_flush_threshold() -> usize {
    4
}

fn default_flush_bytes_threshold() -> usize {
    1024 * 1024
}

/// Log levels for remote-backend logging.
#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum RemoteLogLevel {
    /// Remote logging is disabled.
    #[default]
    #[serde(rename = "disabled")]
    Disabled,

    /// Log periodic summaries — e.g. the cumulative network bytes saved by op-graph caching,
    /// emitted once per additional mebibyte saved.
    #[serde(rename = "basic")]
    Basic,

    /// Log every optimization registration and replay, with per-message sizes.
    #[serde(rename = "full")]
    Full,
}

impl LogLevel for RemoteLogLevel {}
