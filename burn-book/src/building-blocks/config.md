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
#[derive(Config)]
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

## Good practices

By using the Config pattern it is easy to create instances from this
config. Therefore, initialization methods should be implemented on the config struct.

```rust, ignore
impl MyModuleConfig {
    /// Create a module with random weights.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MyModule {
        MyModule {
            linear: LinearConfig::new(self.d_model, self.d_ff).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }

    /// Create a module with a record, for inference and fine-tuning.
    pub fn init_with(&self, record: MyModuleRecord<B>) -> MyModule {
        MyModule {
            linear: LinearConfig::new(
                self.d_model,
                self.d_ff,
            ).init_with(record.linear),
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
