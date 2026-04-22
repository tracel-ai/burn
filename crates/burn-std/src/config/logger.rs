use alloc::{string::String, vec::Vec};
use core::fmt::Display;
use cubecl_common::{
    config::{
        RuntimeConfig,
        logger::{LogLevel, LoggerConfig, LoggerSinks},
    },
    stub::Arc,
};

use super::{autodiff::AutodiffLogLevel, base::BurnConfig, fusion::FusionLogLevel};

static BURN_LOGGER: spin::Mutex<Option<Logger>> = spin::Mutex::new(None);

#[cfg(feature = "std")]
std::thread_local! {
    static LOCAL_CONFIG: std::cell::OnceCell<Arc<BurnConfig>> =
        const { std::cell::OnceCell::new() };
}

/// Returns the current [`BurnConfig`], cached in thread-local storage on native targets.
///
/// On the first call from a given thread this fetches the global config via
/// [`BurnConfig::get`] (which locks a spin mutex) and caches the `Arc` thread-locally.
/// Subsequent calls on the same thread only pay an `Arc` clone. On `no_std` builds this
/// is equivalent to [`BurnConfig::get`].
///
/// Safe because [`BurnConfig::set`] panics after the first read, so the cached snapshot
/// matches the global singleton for the whole program lifetime.
pub fn config() -> Arc<BurnConfig> {
    #[cfg(feature = "std")]
    {
        LOCAL_CONFIG.with(|cell| cell.get_or_init(BurnConfig::get).clone())
    }
    #[cfg(not(feature = "std"))]
    {
        BurnConfig::get()
    }
}

/// Central logging utility for Burn, managing one sink registry shared across subsystems.
#[derive(Debug)]
pub struct Logger {
    sinks: LoggerSinks,
    fusion_index: Vec<usize>,
    autodiff_index: Vec<usize>,
    /// The configuration snapshot the logger was initialized with.
    pub config: Arc<BurnConfig>,
}

impl Default for Logger {
    fn default() -> Self {
        Self::new()
    }
}

impl Logger {
    /// Creates a new `Logger` from the current global `BurnConfig`.
    ///
    /// Note that creating a logger is somewhat expensive because it opens file handles for any
    /// sink configured with a file path.
    pub fn new() -> Self {
        let config = BurnConfig::get();
        let mut sinks = LoggerSinks::new();

        let fusion_index = register_enabled(
            &mut sinks,
            &config.fusion.logger,
            config.fusion.logger.level != FusionLogLevel::Disabled,
        );
        let autodiff_index = register_enabled(
            &mut sinks,
            &config.autodiff.logger,
            config.autodiff.logger.level != AutodiffLogLevel::Disabled,
        );

        Self {
            sinks,
            fusion_index,
            autodiff_index,
            config,
        }
    }

    /// Writes `msg` to all configured fusion sinks.
    pub fn log_fusion<S: Display>(&mut self, msg: &S) {
        self.sinks.log(&self.fusion_index, msg);
    }

    /// Writes `msg` to all configured autodiff sinks.
    pub fn log_autodiff<S: Display>(&mut self, msg: &S) {
        self.sinks.log(&self.autodiff_index, msg);
    }

    /// Returns the current fusion log level.
    pub fn log_level_fusion(&self) -> FusionLogLevel {
        self.config.fusion.logger.level
    }

    /// Returns the current autodiff log level.
    pub fn log_level_autodiff(&self) -> AutodiffLogLevel {
        self.config.autodiff.logger.level
    }
}

fn register_enabled<L: LogLevel>(
    sinks: &mut LoggerSinks,
    config: &LoggerConfig<L>,
    enabled: bool,
) -> Vec<usize> {
    if enabled {
        sinks.register(config)
    } else {
        Vec::new()
    }
}

/// Emit a fusion log message when the configured level is at least `level`.
///
/// The message is only constructed when logging is enabled.
pub fn log_fusion<F>(level: FusionLogLevel, f: F)
where
    F: FnOnce() -> String,
{
    let current = config().fusion.logger.level;
    if current < level {
        return;
    }
    let msg = f();
    let mut guard = BURN_LOGGER.lock();
    let logger = guard.get_or_insert_with(Logger::new);
    logger.log_fusion(&msg);
}

/// Emit an autodiff log message when the configured level is at least `level`.
///
/// The message is only constructed when logging is enabled.
pub fn log_autodiff<F>(level: AutodiffLogLevel, f: F)
where
    F: FnOnce() -> String,
{
    let current = config().autodiff.logger.level;
    if current < level {
        return;
    }
    let msg = f();
    let mut guard = BURN_LOGGER.lock();
    let logger = guard.get_or_insert_with(Logger::new);
    logger.log_autodiff(&msg);
}
