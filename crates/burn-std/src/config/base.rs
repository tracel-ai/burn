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
}
