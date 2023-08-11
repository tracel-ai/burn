# Configs

The `Config` derive lets you define serializable and deserializable configurations or
hyper-parameters for your [modules](#module) or any components.

```rust
use burn::config::Config;

#[derive(Config)]
pub struct PositionWiseFeedForwardConfig {
    pub d_model: usize,
    pub d_ff: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}
```

The `Derive` macro also adds useful methods to your config, such as a builder pattern.

```rust
fn main() {
    let config = PositionWiseFeedForwardConfig::new(512, 2048);
    println!("{}", config.d_model); // 512
    println!("{}", config.d_ff); // 2048
    println!("{}", config.dropout); // 0.1
    let config =  config.with_dropout(0.2);
    println!("{}", config.dropout); // 0.2
}
```
