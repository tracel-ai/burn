use cubecl_common::config::RuntimeConfig;
use cubecl_common::stub::Arc;

use super::autodiff::AutodiffConfig;
use super::fusion::FusionConfig;

/// Static mutex holding the global Burn configuration, initialized as `None`.
static BURN_GLOBAL_CONFIG: spin::Mutex<Option<Arc<BurnConfig>>> = spin::Mutex::new(None);

/// Represents the global configuration for Burn.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BurnConfig {
    /// Configuration for operation fusion.
    #[serde(default)]
    pub fusion: FusionConfig,

    /// Configuration for autodiff.
    #[serde(default)]
    pub autodiff: AutodiffConfig,
}

impl RuntimeConfig for BurnConfig {
    fn storage() -> &'static spin::Mutex<Option<Arc<Self>>> {
        &BURN_GLOBAL_CONFIG
    }

    fn file_names() -> &'static [&'static str] {
        &["burn.toml", "Burn.toml"]
    }

    // Match cubecl-common's `std_io` cfg: only available on platforms where
    // the trait method exists. See cubecl-common's build.rs.
    #[cfg(all(
        feature = "std",
        any(
            target_os = "windows",
            target_os = "linux",
            target_os = "macos",
            target_os = "android"
        )
    ))]
    fn override_from_env(mut self) -> Self {
        use super::fusion::FusionLogLevel;

        if let Ok(val) = std::env::var("BURN_FUSION_LOG") {
            let level = match val.to_ascii_lowercase().as_str() {
                "disabled" | "off" | "0" => FusionLogLevel::Disabled,
                "basic" => FusionLogLevel::Basic,
                "medium" => FusionLogLevel::Medium,
                "full" | "1" => FusionLogLevel::Full,
                _ => self.fusion.logger.level,
            };
            self.fusion.logger.level = level;
            // Default to stderr so tests can see the output via `cargo test -- --nocapture`.
            if level != FusionLogLevel::Disabled {
                self.fusion.logger.stderr = true;
            }
        }

        self
    }
}
