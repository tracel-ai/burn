#![allow(clippy::too_many_arguments)] // To mute derive Config warning
use std::collections::HashMap;

use burn::config::Config;

#[allow(clippy::too_many_arguments)]
#[derive(Debug, PartialEq, Config)]
struct NetConfig {
    n_head: usize,
    n_layer: usize,
    d_model: usize,
    some_float: f64,
    some_int: i32,
    some_bool: bool,
    some_str: String,
    some_list_int: Vec<i32>,
    some_list_str: Vec<String>,
    some_list_float: Vec<f64>,
    some_dict: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use burn_store::pytorch::PytorchReader;

    use super::*;

    #[test]
    fn test_net_config() {
        let config_expected = NetConfig {
            n_head: 2,
            n_layer: 3,
            d_model: 512,
            some_float: 0.1,
            some_int: 1,
            some_bool: true,
            some_str: "hello".to_string(),
            some_list_int: vec![1, 2, 3],
            some_list_str: vec!["hello".to_string(), "world".to_string()],
            some_list_float: vec![0.1, 0.2, 0.3],
            some_dict: {
                let mut map = HashMap::new();
                map.insert("some_key".to_string(), "some_value".to_string());
                map
            },
        };
        let path = "tests/config/weights_with_config.pt";
        let top_level_key = Some("my_config");
        let config: NetConfig = PytorchReader::load_config(path, top_level_key).unwrap();

        assert_eq!(config, config_expected);
    }
}
