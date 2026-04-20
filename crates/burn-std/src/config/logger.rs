use serde::Serialize;
use serde::de::DeserializeOwned;

#[cfg(feature = "std")]
use std::path::PathBuf;

/// Configuration for a log sink, parameterized by a subsystem-specific log level.
///
/// Multiple sinks can be enabled at the same time (e.g. both `stdout` and a file).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct LoggerConfig<L: LogLevel> {
    /// Path to the log file, if file logging is enabled (requires the `std` feature).
    #[serde(default)]
    #[cfg(feature = "std")]
    pub file: Option<PathBuf>,

    /// Whether to append to the log file (true) or overwrite it (false). Defaults to true.
    #[serde(default = "append_default")]
    pub append: bool,

    /// Whether to log to standard output.
    #[serde(default)]
    pub stdout: bool,

    /// Whether to log to standard error.
    #[serde(default)]
    pub stderr: bool,

    /// Optional crate-level logging configuration (e.g., info, debug, trace).
    #[serde(default)]
    pub log: Option<LogCrateLevel>,

    /// The verbosity level for this subsystem.
    #[serde(default)]
    pub level: L,
}

impl<L: LogLevel> Default for LoggerConfig<L> {
    fn default() -> Self {
        Self {
            #[cfg(feature = "std")]
            file: None,
            append: true,
            stdout: false,
            stderr: false,
            log: None,
            level: L::default(),
        }
    }
}

/// Log levels forwarded to the `log` crate.
#[derive(
    Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize, Hash, PartialEq, Eq,
)]
pub enum LogCrateLevel {
    /// Informational messages.
    #[default]
    #[serde(rename = "info")]
    Info,

    /// Debugging messages.
    #[serde(rename = "debug")]
    Debug,

    /// Trace-level messages.
    #[serde(rename = "trace")]
    Trace,
}

fn append_default() -> bool {
    true
}

/// Trait for types usable as a subsystem-specific log level.
pub trait LogLevel:
    DeserializeOwned + Serialize + Clone + Copy + core::fmt::Debug + Default
{
}
