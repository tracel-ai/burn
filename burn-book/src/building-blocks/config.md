# Config

When writing scientific code, you normally have a lot of values that are set, and Deep Learning is
no exception. Python has the possibility to define default parameters for functions, which helps
improve the developer experience. However, this has the downside of potentially breaking your code
when upgrading to a new version, as the default values might change without your knowledge, making
debugging very challenging.

With that in mind, we came up with the Config system. It's a simple Rust derive that you can apply
to your types, allowing you to define default values with ease. Additionally, all configs can be
serialized, reducing potential bugs when upgrading versions and improving reproducibility.

```rust , ignore
use burn::config::Config;

#[derive(Config)]
pub struct MyModuleConfig {
    d_model: usize,
    d_ff: usize,
    #[config(default = 0.1)]
    dropout: f64,
}
```

The derive also adds useful `with_` methods for every attribute of your config, similar to a builder
pattern, along with a `save` method.

```rust, ignore
fn main() {
    let config = MyModuleConfig::new(512, 2048);
    println!("{}", config.d_model); // 512
    println!("{}", config.d_ff); // 2048
    println!("{}", config.dropout); // 0.1
    let config =  config.with_dropout(0.2);
    println!("{}", config.dropout); // 0.2

    config.save("config.json").unwrap();
}
```

## Documentation

The Config derive macro automatically generates documentation for your config structs. You can add documentation comments to your fields to provide more information about their purpose and usage. The documentation will be included in the generated `new` function and builder methods.

```rust, ignore
#[derive(Config)]
pub struct MyModuleConfig {
    /// The dimension of the model.
    d_model: usize,
    /// The dimension of the feed-forward network.
    d_ff: usize,
    /// The dropout probability.
    #[config(default = 0.1)]
    dropout: f64,
}
```

The documentation will be used to generate helpful descriptions in the `new` function:

```rust, ignore
/// Create a new instance of the config.
///
/// Fields:
/// - `d_model`: The dimension of the model.
/// - `d_ff`: The dimension of the feed-forward network.
/// - `dropout`: The dropout probability.
/// Default: 0.1
pub fn new(
    d_model: usize,
    d_ff: usize,
) -> Self {
    // ...
}
```

And in the builder methods:

```rust, ignore
/// Set the dropout probability.
///
/// Defaults to `0.1`.
pub fn with_dropout(mut self, dropout: f64) -> Self {
    self.dropout = dropout;
    self
}
```

## Good practices

By using the config type it is easy to create new module instances. The initialization method should
be implemented on the config type with the device as argument.

```rust, ignore
impl MyModuleConfig {
    /// Create a module on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MyModule {
        MyModule {
            linear: LinearConfig::new(self.d_model, self.d_ff).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}
```

Then we could add this line to the above `main`:

```rust, ignore
use burn::backend::Wgpu;
let device = Default::default();
let my_module = config.init::<Wgpu>(&device);
```
