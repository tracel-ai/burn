# Inference

Now that we have trained our model, the next natural step is to use it for inference.

You need two things in order to load weights for a model: the model's record and the model's config.
Since parameters in Burn are lazy initialized, no allocation and GPU/CPU kernels are executed by the
`ModelConfig::init` function. The weights are initialized when used for the first time, therefore
you can safely use `config.init(device).load_record(record)` without any meaningful performance
cost. Let's create a simple `infer` method in a new file `src/inference.rs` which we will use to
load our trained model.

```rust , ignore
# use burn::{
#     config::Config,
#     data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
#     module::Module,
#     record::{CompactRecorder, Recorder},
#     tensor::backend::Backend,
# };
# 
# use crate::{data::MnistBatcher, training::TrainingConfig};
# 
pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MnistItem) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    let label = item.label;
    let batcher = MnistBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
}
```

The first step is to load the configuration of the training to fetch the correct model
configuration. Then we can fetch the record using the same recorder as we used during training.
Finally we can init the model with the configuration and the record. For simplicity we can use the
same batcher used during the training to pass from a MnistItem to a tensor.

By running the infer function, you should see the predictions of your model!

Add the call to `infer` to the `main.rs` file after the `train` function call:

```rust , ignore
# mod data;
# mod inference;
# mod model;
# mod training;
# 
# use crate::{model::ModelConfig, training::TrainingConfig};
# use burn::{
#     backend::{Autodiff, Wgpu},
#     data::dataset::Dataset,
#     optim::AdamConfig,
# };
# 
# fn main() {
#     type MyBackend = Wgpu<f32, i32>;
#     type MyAutodiffBackend = Autodiff<MyBackend>;
# 
#     let device = burn::backend::wgpu::WgpuDevice::default();
#     let artifact_dir = "/tmp/guide";
#     crate::training::train::<MyAutodiffBackend>(
#         artifact_dir,
#         TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
#         device.clone(),
#     );
    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
# }
```

The number `42` is the index of the image in the MNIST dataset. You can explore and verify them using
this [MNIST viewer](https://observablehq.com/@davidalber/mnist-viewer).
