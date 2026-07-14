use cubecl_common::config::logger::{LogLevel, LoggerConfig};

/// Configuration for autodiff in Burn.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AutodiffConfig {
    /// Logger configuration for autodiff logs.
    #[serde(default)]
    pub logger: LoggerConfig<AutodiffLogLevel>,
}

/// Log levels for autodiff logging.
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
pub enum AutodiffLogLevel {
    /// Autodiff logging is disabled.
    #[default]
    #[serde(rename = "disabled")]
    Disabled,

    /// Log backward graph size and the checkpoint strategy applied per forward pass.
    #[serde(rename = "basic")]
    Basic,

    /// Additionally log each tensor that gets checkpointed or recomputed.
    #[serde(rename = "medium")]
    Medium,

    /// Log every graph node traversal and recomputation event.
    #[serde(rename = "full")]
    Full,
}

impl LogLevel for AutodiffLogLevel {}
