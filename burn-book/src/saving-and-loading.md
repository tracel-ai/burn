# Saving and Loading Models

Saving your trained machine learning model to disk is quite easy, no matter the output format you choose. As mentioned in the [Record](./building-blocks/record.md) section, different formats are supported to serialize/deserialize models. By default, we use the `NamedMpkFileRecorder` which uses the [MessagePack](https://msgpack.org/) binary serialization format with the help of [smp_serde](https://docs.rs/rmp-serde/).

```rust, ignore
// Save model in MessagePack format with full precision
let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
model
  .save_file(model_path, &recorder)
  .expect("Should be able to save the model");
```

Note that the file extension is automatically handled by the recorder depending on the one you choose. Therefore, only the file path and base name should be provided.

Now that you have a trained model saved to your disk, you can easily load it in a similar fashion.

```rust, ignore
// Load model in full precision from MessagePack file
let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
model
  .load_file(model_path, &recorder)
  .expect("Should be able to load the model weights from the provided file");
```

**Note:** models can be saved in different output formats, just make sure you are using the correct recorder type when loading the saved model. Type conversion between different precision settings is automatically handled, but formats are not interchangeable. A model can be loaded from one format and saved to another format, just as long as you load it back with the new recorder type afterwards.

## Custom Initialization from Recorded Weights

While the first approach is very straightforward, it does require the model to already be initialized. If instead you would like to skip the initialization and directly load the weights into the modules of your model, you can create a new initialization function. Let's take the following model definition as a simple example.

```rust, ignore
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear_in: Linear<B>,
    linear_out: Linear<B>,
    activation: ReLU,
}
```

Similar to the [basic workflow inference example](../basic-workflow/inference.md), we can define a new initialization function which initializes the different parts of our model with the record values.

```rust, ignore
impl<B: Backend> Model<B> {
    /// Returns the initialized model using the recorded weights.
    pub fn init_with(record: ModelRecord<B>) -> Model<B> {
        Model {
            linear_in: LinearConfig::new(10, 64).init_with(record.linear_in),
            linear_out: LinearConfig::new(64, 2).init_with(record.linear_out),
            activation: ReLU::new(),
        }
    }

    /// Returns the dummy model with randomly initialized weights.
    pub fn new(device: &Device<B>) -> Model<B> {
        let l1 = LinearConfig::new(10, 64).init(device);
        let l2 = LinearConfig::new(64, 2).init(device);
        Model {
            linear_in: l1,
            linear_out: l2,
            activation: ReLU::new(),
        }
    }
}``
```

Now, let's save a model that we can load later. In the following snippets, we use `type Backend = NdArray<f32>` but you can use whatever backend you like.

```rust, ignore
// Create a dummy initialized model to save
let model = Model::<Backend>::new(&Default::default());

// Save model in MessagePack format with full precision
let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
model
    .save_file(model_path, &recorder)
    .expect("Should be able to save the model");
```

Afterwards, the model can just as easily be loaded from the record saved on disk.

```rust, ignore
// Load model record on the backend's default device
let record: ModelRecord<Backend> = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
    .load(model_path.into())
    .expect("Should be able to load the model weights from the provided file");

// Directly initialize a new model with the loaded record/weights
let model = Model::init_with(record);
```