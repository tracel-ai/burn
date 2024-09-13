use burn::config::{config_to_json, Config};
use burn_core as burn;

#[derive(Config, Debug, PartialEq, Eq)]
pub struct TestEmptyStructConfig {}

#[derive(Config, Debug, PartialEq)]
pub struct TestStructConfig {
    int: i32,
    #[config(default = 2)]
    int_default: i32,
    float: f32,
    #[config(default = 2.0)]
    float_default: f32,
    string: String,
    other_config: TestEmptyStructConfig,
}

#[derive(Config, Debug, PartialEq)]
pub enum TestEnumConfig {
    None,
    Single(f32),
    Multiple(f32, String),
    Named { first: f32, second: String },
}

#[cfg(feature = "std")]
#[inline(always)]
fn file_path(file_name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(file_name)
}

#[cfg(feature = "std")]
#[test]
fn struct_config_should_impl_serde() {
    let config = TestStructConfig::new(2, 3.0, "Allow".to_string(), TestEmptyStructConfig::new());
    let file_path = file_path("test_struct_config.json");

    config.save(&file_path).unwrap();

    let config_loaded = TestStructConfig::load(&file_path).unwrap();
    assert_eq!(config, config_loaded);
}

#[test]
fn struct_config_should_impl_clone() {
    let config = TestStructConfig::new(2, 3.0, "Allow".to_string(), TestEmptyStructConfig::new());
    assert_eq!(config, config.clone());
}

#[test]
fn struct_config_should_impl_display() {
    let config = TestStructConfig::new(2, 3.0, "Allow".to_string(), TestEmptyStructConfig::new());
    assert_eq!(burn::config::config_to_json(&config), config.to_string());
}

#[cfg(feature = "std")]
#[test]
fn enum_config_no_value_should_impl_serde() {
    let config = TestEnumConfig::None;
    let file_path = file_path("test_enum_no_value_config.json");

    config.save(&file_path).unwrap();

    let config_loaded = TestEnumConfig::load(&file_path).unwrap();
    assert_eq!(config, config_loaded);
}

#[cfg(feature = "std")]
#[test]
fn enum_config_one_value_should_impl_serde() {
    let config = TestEnumConfig::Single(42.0);
    let file_path = file_path("test_enum_one_value_config.json");

    config.save(&file_path).unwrap();

    let config_loaded = TestEnumConfig::load(&file_path).unwrap();
    assert_eq!(config, config_loaded);
}

#[cfg(feature = "std")]
#[test]
fn enum_config_multiple_values_should_impl_serde() {
    let config = TestEnumConfig::Multiple(42.0, "Allow".to_string());
    let file_path = file_path("test_enum_multiple_values_config.json");

    config.save(&file_path).unwrap();

    let config_loaded = TestEnumConfig::load(&file_path).unwrap();
    assert_eq!(config, config_loaded);
}

#[test]
fn enum_config_should_impl_clone() {
    let config = TestEnumConfig::Multiple(42.0, "Allow".to_string());
    assert_eq!(config, config.clone());
}

#[test]
fn enum_config_should_impl_display() {
    let config = TestEnumConfig::Multiple(42.0, "Allow".to_string());
    assert_eq!(burn::config::config_to_json(&config), config.to_string());
}

#[test]
fn struct_config_can_load_binary() {
    let config = TestStructConfig::new(2, 3.0, "Allow".to_string(), TestEmptyStructConfig::new());

    let binary = config_to_json(&config).as_bytes().to_vec();

    let config_loaded = TestStructConfig::load_binary(&binary).unwrap();
    assert_eq!(config, config_loaded);
}
