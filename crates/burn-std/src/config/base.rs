use cubecl_common::config::RuntimeConfig;
use cubecl_common::stub::Arc;

use super::autodiff::AutodiffConfig;
use super::fusion::FusionConfig;
use super::remote::RemoteConfig;

/// Static mutex holding the global Burn configuration, initialized as `None`.
static BURN_GLOBAL_CONFIG: spin::Mutex<Option<Arc<BurnConfig>>> = spin::Mutex::new(None);

/// Represents the global configuration for Burn.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BurnConfig {
    /// Configuration for operation fusion.
    #[serde(default)]
    fusion: FusionConfig,

    /// Configuration for autodiff.
    #[serde(default)]
    autodiff: AutodiffConfig,

    /// Configuration for the remote backend.
    #[serde(default)]
    remote: RemoteConfig,
}

impl BurnConfig {
    /// Returns a reference to the operation-fusion configuration.
    pub fn fusion(&self) -> &FusionConfig {
        &self.fusion
    }

    /// Returns a reference to the autodiff configuration.
    pub fn autodiff(&self) -> &AutodiffConfig {
        &self.autodiff
    }

    /// Returns a reference to the remote-backend configuration.
    pub fn remote(&self) -> &RemoteConfig {
        &self.remote
    }
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
        use super::remote::RemoteLogLevel;

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

        if let Ok(val) = std::env::var("BURN_FUSION_MAX_EXPLORATIONS")
            && let Ok(n) = val.parse::<usize>()
        {
            self.fusion.beam_search.max_explorations = Some(n);
        }

        if let Ok(val) = std::env::var("BURN_REMOTE_LOG") {
            let level = match val.to_ascii_lowercase().as_str() {
                "disabled" | "off" | "0" => RemoteLogLevel::Disabled,
                "basic" | "1" => RemoteLogLevel::Basic,
                "full" | "2" => RemoteLogLevel::Full,
                _ => self.remote.logger.level,
            };
            self.remote.logger.level = level;
            if level != RemoteLogLevel::Disabled {
                self.remote.logger.stderr = true;
            }
        }

        self
    }
}
