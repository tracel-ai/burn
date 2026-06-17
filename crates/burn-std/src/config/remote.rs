use cubecl_common::config::logger::{LogLevel, LoggerConfig};

/// Configuration for the remote backend.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RemoteConfig {
    /// Logger configuration for remote-backend logs (e.g. the bytes-saved metric of the
    /// fusion / op-graph caching feature).
    #[serde(default)]
    pub logger: LoggerConfig<RemoteLogLevel>,
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
