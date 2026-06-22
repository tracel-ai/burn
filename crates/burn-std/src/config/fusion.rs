use cubecl_common::config::logger::{LogLevel, LoggerConfig};

/// Configuration for operation fusion in Burn.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FusionConfig {
    /// Logger configuration for fusion logs.
    #[serde(default)]
    pub logger: LoggerConfig<FusionLogLevel>,

    /// Beam search configuration used when exploring fusion opportunities.
    #[serde(default)]
    pub beam_search: BeamSearchConfig,

    /// Maximum number of operations in a single client-cached graph (router graph caching).
    ///
    /// The greedy graph fuser keeps accumulating ops into one cached graph; once it reaches this
    /// many ops the graph is closed and dispatched. `None` (the default) sets no size cap, leaving
    /// [`growth_patience`](Self::growth_patience) to decide when a graph stops being worth growing;
    /// set a value to also bound the per-graph memory and the cost of building and replaying any
    /// single graph.
    #[serde(default)]
    pub max_graph_size: Option<usize>,

    /// Close the current graph once its fusion score hasn't reached a new maximum for this many
    /// consecutive ops (router graph caching).
    ///
    /// The fuser scores the accumulated ops as they grow and tracks the best score so far; once
    /// this many ops have been added without beating it, the graph has stopped getting more worth
    /// caching and is closed. Higher values keep growing the graph longer in search of a better
    /// score.
    #[serde(default = "default_growth_patience")]
    pub growth_patience: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            logger: LoggerConfig::default(),
            beam_search: BeamSearchConfig::default(),
            max_graph_size: None,
            growth_patience: default_growth_patience(),
        }
    }
}

fn default_growth_patience() -> usize {
    32
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

    /// Stop exploring new optimizations after this many explorations (per stream).
    ///
    /// Each cache miss runs the (relatively expensive) optimizer to build a new optimization. A
    /// graph whose relative form changes every step never caches, so it re-explores forever. Once
    /// this cap is reached, cache-missing segments execute *unfused* (no optimizer work, nothing
    /// added to the cache) instead. Cache *hits* are unaffected, so already-cached stable graphs
    /// keep replaying. `None` (the default) never disables exploration.
    #[serde(default)]
    pub max_explorations: Option<usize>,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            max_blocks: default_max_blocks(),
            max_explorations: None,
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
