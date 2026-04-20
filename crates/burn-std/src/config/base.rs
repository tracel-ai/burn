use alloc::sync::Arc;
use cubecl_common::config::RuntimeConfig;

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

    /// Configuration for the CubeCL runtime.
    ///
    /// Propagated to CubeCL's global config on the first call to
    /// [`crate::config::config`], so a single `burn.toml` can hold both Burn and CubeCL
    /// settings under `[cubecl.autotune]`, `[cubecl.compilation]`, etc.
    #[cfg(feature = "cubecl")]
    #[serde(default)]
    pub cubecl: cubecl::config::CubeClRuntimeConfig,
}

impl RuntimeConfig for BurnConfig {
    fn storage() -> &'static spin::Mutex<Option<Arc<Self>>> {
        &BURN_GLOBAL_CONFIG
    }

    fn file_names() -> &'static [&'static str] {
        &["burn.toml", "Burn.toml"]
    }
}
