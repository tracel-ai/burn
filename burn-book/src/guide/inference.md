# Inference

Now that we have trained our model, we may want to use the model for inference.
To load a model ready for inference, you normally prefer to directly load the weights in the model without having the initiaize a model with something like xavier normal only to then overriden everything with the saved weights.
For efficency, let's create a new initialization function receiving the record as input.


```rust, ignore
impl ModelConfig {
    /// Returns the initialized model using the recorded weights.
    pub fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init_with(record.conv1),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init_with(record.conv2),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: ReLU::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init_with(record.linear1),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes)
                .init_with(record.linear2),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}
```

Note that the record is generated automatically with the `Module` trait.
It allows you to load module state without having to deal with featching the correct type in a dictionaly or something manual.
Everything is validated when loading the model with the record.

Now let's create a simple infer method that shows how to load a trained model.

```rust, ignore
pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MNISTItem) {
    let config =
        TrainingConfig::load(&format!("{artifact_dir}/config.json")).expect("A config exists");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into())
        .expect("Failed to save trained model");

    let model = config.model.init_with::<B>(record).to_device(&device);

    let label = item.label;
    let batcher = MNISTBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
}
```

The first step is to load the configuration of the training to fetch the correct model configuration.
Then we can fetch the record using the same recorder that was used during training.
Finaly we can init the model with the config and the record before sending it to the wanted device for inference.

For simplicity we can use the same batcher used during the training to pass from a MNISTItem to a tensor.
