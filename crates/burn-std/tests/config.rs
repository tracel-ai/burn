/// Tests for loading `burn.toml` from disk.
///
/// The `cubecl` feature toggles whether `BurnConfig` has a `cubecl` sub-config. We still
/// want `burn.toml` files that contain `[cubecl.*]` sections to load cleanly either way —
/// serde's default behaviour is to ignore unknown top-level fields, and these tests lock
/// that in.
#[cfg(feature = "std")]
mod config_loading {
    use burn_std::config::autodiff::AutodiffLogLevel;
    use burn_std::config::fusion::FusionLogLevel;
    use burn_std::config::{BurnConfig, RuntimeConfig};
    use std::io::Write;
    use tempfile::NamedTempFile;

    const EMPTY_TOML: &str = "";

    const MINIMAL_TOML: &str = r#"
[fusion.beam_search]
max_blocks = 3
"#;

    const FULL_TOML: &str = r#"
[fusion.beam_search]
max_blocks = 10

[fusion.logger]
level = "medium"
stdout = true

[autodiff.logger]
level = "basic"

[cubecl.autotune]
level = "full"

[cubecl.compilation.logger]
level = "full"
stdout = true
"#;

    fn write_toml(content: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("create temp toml");
        file.write_all(content.as_bytes()).expect("write toml");
        file.flush().expect("flush toml");
        file
    }

    #[test]
    fn load_empty_toml_uses_defaults() {
        let file = write_toml(EMPTY_TOML);
        let config = BurnConfig::from_file_path(file.path()).expect("parse empty toml");

        assert_eq!(config.fusion.beam_search.max_blocks, 5);
        assert_eq!(config.fusion.logger.level, FusionLogLevel::Disabled);
        assert_eq!(config.autodiff.logger.level, AutodiffLogLevel::Disabled);
    }

    #[test]
    fn load_minimal_toml_overrides_defaults() {
        let file = write_toml(MINIMAL_TOML);
        let config = BurnConfig::from_file_path(file.path()).expect("parse minimal toml");

        assert_eq!(config.fusion.beam_search.max_blocks, 3);
        // Untouched sections keep defaults.
        assert_eq!(config.autodiff.logger.level, AutodiffLogLevel::Disabled);
    }

    /// The important case: a single `burn.toml` with both Burn and CubeCL sections must
    /// load whether or not the `cubecl` feature is enabled on `burn-std`.
    #[test]
    fn load_full_toml_ignores_cubecl_when_feature_off() {
        let file = write_toml(FULL_TOML);
        let config = BurnConfig::from_file_path(file.path()).expect("parse full toml");

        assert_eq!(config.fusion.beam_search.max_blocks, 10);
        assert_eq!(config.fusion.logger.level, FusionLogLevel::Medium);
        assert!(config.fusion.logger.stdout);
        assert_eq!(config.autodiff.logger.level, AutodiffLogLevel::Basic);
    }
}
