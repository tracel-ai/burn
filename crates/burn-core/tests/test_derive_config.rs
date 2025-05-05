use burn::config::{Config, config_to_json};
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

/// A test struct for verifying documentation generation.
#[derive(Config, Debug)]
pub struct TestDocConfig {
    /// This is a required integer field.
    pub required_int: i32,

    /// This is an optional string field.
    pub optional_string: Option<String>,

    /// This is a field with a default value.
    #[config(default = 42)]
    pub default_int: i32,

    /// This is a field with a default string value.
    #[config(default = "\"hello\"")]
    pub default_string: String,
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

#[test]
fn test_config_documentation() {
    let config = TestDocConfig::new(123);

    // Get the type information using syn
    let source = quote::quote! {
        #[derive(Config, Debug)]
        pub struct TestDocConfig {
            /// This is a required integer field.
            pub required_int: i32,

            /// This is an optional string field.
            pub optional_string: Option<String>,

            /// This is a field with a default value.
            #[config(default = 42)]
            pub default_int: i32,

            /// This is a field with a default string value.
            #[config(default = "\"hello\"")]
            pub default_string: String,
        }
    };

    let ast: syn::DeriveInput = syn::parse2(source).unwrap();
    let fields = match ast.data {
        syn::Data::Struct(data) => data.fields,
        _ => panic!("Expected struct"),
    };

    // Verify field documentation
    let fields: Vec<_> = fields.into_iter().collect();
    assert_eq!(fields.len(), 4);

    // Check required field
    let required_docs = fields[0]
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .count();
    assert_eq!(required_docs, 1);

    // Check optional field
    let optional_docs = fields[1]
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .count();
    assert_eq!(optional_docs, 1);

    // Check default field
    let default_docs = fields[2]
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .count();
    assert_eq!(default_docs, 1);

    // Check default string field
    let default_string_docs = fields[3]
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .count();
    assert_eq!(default_string_docs, 1);

    // Test builder methods
    let config = config.clone();
    let config = config.with_default_int(100);
    let config = config.with_default_string("world".to_string());
    let _config = config.with_optional_string(Some("optional".to_string()));
}
