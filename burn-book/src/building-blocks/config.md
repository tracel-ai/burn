# Config

When writing scientific code, you normally have a lot of values that are set, and Deep Learning is
no exception. Python has the possibility to define default parameters for functions, which helps
improve the developer experience. However, this has the downside of potentially breaking your code
when upgrading to a new version, as the default values might change without your knowledge, making
debugging very challenging.

With that in mind, we came up with the Config system. It's a simple Rust derive that you can apply
to your types, allowing you to define default values with ease. Additionally, all configs can be
serialized, reducing potential bugs when upgrading versions and improving reproducibility.

```rust, ignore
#[derive(Config)]
use burn::config::Config;

#[derive(Config)]
pub struct MyConfig {
    d_model: usize,
    d_ff: usize,
    #[config(default = 0.1)]
    dropout: f64,
}
```

The derive also adds useful methods to your config, similar to a builder pattern.

```rust
fn main() {
    let config = MyConfig::new(512, 2048);
    println!("{}", config.d_model); // 512
    println!("{}", config.d_ff); // 2048
    println!("{}", config.dropout); // 0.1
    let config =  config.with_dropout(0.2);
    println!("{}", config.dropout); // 0.2
}
```
