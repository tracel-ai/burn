# Model

The first step is to create a project and add the different Burn dependencies.
In a `Cargo.toml` file, add the `burn`, `burn-wgpu`, `burn-dataset`, `burn-autodiff` and `burn-train`.
Note that the `serde` dependency is necessary for serialization and is mandatory for the time being.

```toml
[package]
edition = "2021"
name = "My first Burn model"

[dependencies]
burn = "0.9"
burn-wgpu = "0.9"
burn-dataset = "0.9"
burn-autodiff = "0.9"
burn-train = "0.9"

# Serialization
serde = "1"
```

Our goal will be to create a basic convolutional neural network used for image classification. We will keep the model simple by using two convolution layers followed by two linear layers, some pooling and ReLU activations.
We will also use dropout to improve training performance.

Let us start by creating a model in a file `model.rs`.

```rust , ignore
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
This trait will make your module both trainable and (de)serializable while adding related functionalities.
Like other attributes often used in Rust, such as `Clone`, `PartialEq` or `Debug`, each field within the struct must also implement the `Module` trait.

2. Note that the struct is generic over the `Backend` trait.
The backend trait abstracts the underlying low level implementations of tensor operations, allowing your new model to run on any backend.
Contrary to other frameworks, the backend abstraction isn't determined by a compilation flag or a device type.
This is important because you can extend the functionalities of a specific backend (which will be covered in the more advanced sections of this book), and it allows for an innovative autodiff system.
You can also change backend during runtime, for instance to compute training metrics on a cpu backend while using a gpu one only to train the model. 
In our example, the backend in use will be determined later on.

Next, we need to instantiate the model for training.

```rust , ignore
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
This allows you to define default values for your network, thanks to the `Config` attribute.
The benefit of this attribute is that it makes the configuration serializable, enabling you to painlessly save your model hyperparameters, enhancing your experimentation process.
Note that a constructor will automatically be generated for your configuration, which will take as input values for the parameter which do not have default values: `let config = ModelConfig::new(num_classes, hidden_size);`.
The default values can be overridden easily with builder-like methods: (e.g `config.with_dropout(0.2);`)

The first implementation block is related to the initialization method.
As we can see, all fields are set using the configuration of the corresponding neural network underlying module.
In this specific case, we have chosen to expand the tensor channels from 1 to 8 with the first layer, then from 8 to 16 with the second layer, using a kernel size of 3 on all dimensions.
We also use the adaptive average pooling module to reduce the dimensionality of the images to an 8 by 8 matrix, which we will flatten in the forward pass to have a 1024 (16 * 8 * 8) resulting tensor.

Now let's see how the forward pass is defined.

```rust , ignore
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

For former PyTorch users, this might feel very intuitive, as each module is directly incorporated into the code using an eager API.
Note that no abstraction is imposed for the forward method. You are free to define multiple forward functions with the names of your liking.
Most of the neural network modules already built with Burn use the `forward` nomenclature, simply because it is standard in the field.

Similar to neural network modules, the `Tensor` struct given as a parameter also takes the Backend trait as a generic argument, alongside its rank.
Even if it is not used in this specific example, it is possible to add the kind of the tensor as a third generic argument.

```rust , ignore
Tensor<B, 3> // Float tensor (default)
Tensor<B, 3, Float> // Float tensor (explicit)
Tensor<B, 3, Int> // Int tensor
Tensor<B, 3, Bool> // Bool tensor
```

Note that the specific element type, such as `f16`, `f32` and the likes, will be defined later with the backend.
