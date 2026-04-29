use cubecl_common::config::logger::{LogLevel, LoggerConfig};

/// Configuration for operation fusion in Burn.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FusionConfig {
    /// Logger configuration for fusion logs.
    #[serde(default)]
    pub logger: LoggerConfig<FusionLogLevel>,

    /// Beam search configuration used when exploring fusion opportunities.
    #[serde(default)]
    pub beam_search: BeamSearchConfig,
}

/// Beam search configuration controlling how the fusion optimizer explores independent blocks
/// of operations.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BeamSearchConfig {
    /// Maximum number of independent blocks explored during the fusion search.
    ///
    /// Higher values can find better fusion opportunities at the cost of more cache misses
    /// in the fusion cache.
    #[serde(default = "default_max_blocks")]
    pub max_blocks: usize,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            max_blocks: default_max_blocks(),
        }
    }
}

fn default_max_blocks() -> usize {
    5
}

/// Log levels for fusion logging.
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
pub enum FusionLogLevel {
    /// Fusion logging is disabled.
    #[default]
    #[serde(rename = "disabled")]
    Disabled,

    /// Log the final execution strategy selected per stream (single vs composed).
    #[serde(rename = "basic")]
    Basic,

    /// Log block merge/split decisions and cache hit/miss events.
    #[serde(rename = "medium")]
    Medium,

    /// Log every registration, rejection and scoring decision.
    #[serde(rename = "full")]
    Full,
}

impl LogLevel for FusionLogLevel {}
