# Model

The first step is to create a project and adding the different Burn dependencies.
In a `Cargo.toml` add the `burn`, `burn-autodiff`, `burn-train` and `burn-ndarray` as we will use the `ndarray` backend for this example.
Note that serde is necessary for serialization and is mandatory for now.

```toml
[package]
edition = "2021"
name = "My package"

[dependencies]
burn = "0.8"
burn-ndarray = "0.8"
burn-dataset = "0.8"
burn-autodiff = "0.8"
burn-train = "0.8"

# Serialization
serde = "1"
```

Our goal is to create a basic convolutional neural network used for image classification.
Let's keep the model simple by using two convolution layers followed by two linear layers, some pooling and relu activation.
We will use dropout to improve training performance.
We will start by creating a model in a file `model.rs`.

```rust,ignore
// Import required for the model.rs file
use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, ReLU,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: ReLU,
}
```

There are two major things going on in this code sample.

1. You can create a deep learning module with the `#[derive(Module)]` attribute on top of a struct.
This will generate the necessary code so that the struct implements the `Module` trait.
This trait will make your module trainable, (de)serializable and add related functionlities.
Like other attributes often used in Rust, such as `Clone`, `PartialEq` or `Debug`, each field in the struct must also implement the `Module` trait.

2. Note that the struct is generic over the `Backend` trait.
The backend trait abstract the underlying low level implementations of tensor operations allowing your new model to run on any backend.
Contrary to other frameworks, the backend abstraction isn't a compilation flag or determined by a device type.
This is important since you can extend the functionlities of a specific backend (advanced) and it allows for an innovative autodiff system.
You can also change backend during runtime, such as a cpu backend to compute training metrics and another gpu backend that only trains the model.
We will determined the backend in used for this example later on.

Next, we need to intaciate the model for training.

```rust,ignore
#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: ReLU::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}
```

When creating a custom neural network module, it is often a good idea to create a config alongside the model struct.
This allow you to defined default values for your network thanks to the `Config` attribute.
The benefit of the `Config` attribute is that your configuration is now serializable, enabling you to painlessly save your model hyper-parameter enhancing your experimentation exxperience.
Note that a constructor will be created for your configuration with the parameter without default as input: `let config = ModelConfig::new(num_classes, hidden_size);`.
The defautl values can be overriden easily with builder like methods: (e.g `config.with_dropout(0.2);`)

The first implementation block is related to the initialization method.
As we can see, each field is set using the configuration of each neural network underlying module.
In this specific case, we chose to expand the tensor channels from 1 to 8 with the first layer, and from 8 to 16 from the second layer using kernel size of 3 for all dimensions.
We also use the adaptive avg pooling module to reduce the dimensionality of the images to an 8 by 8 matrix, which we will flatten in the forward pass to have a 1024 (16 * 8 * 8) resulting tensor.
Now let's see how the forward pass is defined.

```rust, ignore
impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, 1, height, width]);


        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
    }
}
```

For PyTorch users, this might feel very intuitive as each module is used directly into the code with an eager API.
Note that no abtraction is forced for the forward method, you are free to define multiple forward functions with the name of your liking.
Most of the neural network modules already built with burn uses the `forward` nomeclature since it's pretty standard in the field.

Similar to neural network modules, the `Tensor` struct also takes the Backend trait as generic argument alongside its rank.
Even if it's not used in this specific example, you can add as a third generic argument, the kind of the tensor.

```rust, ignore
Tensor<B, 3> // Float tensor (default)
Tensor<B, 3, Float> // Float tensor (explicit)
Tensor<B, 3, Int> // Int tensor
Tensor<B, 3, Bool> // Bool tensor
```

Note that the specific element type used by the backend, such as `f16`, `f32` and the likes, is defined later.
