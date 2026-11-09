# Config

For process-wide settings such as logging and fusion in `burn.toml`, see
[Runtime Configuration](../advanced/runtime-configuration.md).

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

#[derive(Config, Debug)]
pub struct MyModuleConfig {
    d_model: usize,
    d_ff: usize,
    #[config(default = 0.1)]
    dropout: f64,
}
```

The derive also adds `with_` methods for optional fields and fields with explicit defaults, similar
to a builder pattern, along with a `save` method.

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

## Converting arguments

Add `#[config(into)]` to a field to let its generated constructor argument or setter accept
`impl Into<FieldType>`. For example, a `String` field can accept `&str`, and a `Vec<usize>` field
can accept an array:

```rust, ignore
use burn::config::Config;

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(into)]
    name: String,
    width: usize,
    #[config(default = "Vec::new()", into)]
    hidden_sizes: Vec<usize>,
}

let config = ModelConfig::new("model", 128).with_hidden_sizes([64, 32]);
```

You can also write `#[config(into)]` and `#[config(default = ...)]` as separate attributes.
The field's stored type, default expression and serialized representation stay the same.
For an `Option<T>` field, the input must convert to the whole `Option<T>`; conversions are not
automatically applied to its contents. For example, `Into<Option<String>>` accepts `String`,
`Some(String)` or `None`, but not `&str`.

Opt in where conversion improves call sites. Unannotated fields retain their exact argument
types and type inference, so `width` above still accepts an unsuffixed integer literal. Opting in
can require removing existing `.into()` calls or adding types to `.collect()` and
`Default::default()` calls.

## Good practices

By using the config type it is easy to create new module instances. The initialization method should
be implemented on the config type with the device as argument.

```rust, ignore
impl MyModuleConfig {
    /// Create a module on the given device.
    pub fn init(&self, device: &Device) -> MyModule {
        MyModule {
            linear: LinearConfig::new(self.d_model, self.d_ff).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}
```

Then we could add this line to the above `main`:

```rust, ignore
use burn::tensor::Device;
let device = Device::wgpu(Default::default());
let my_module = config.init(&device);
```
