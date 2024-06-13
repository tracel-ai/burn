# Backend

We have effectively written most of the necessary code to train our model. However, we have not
explicitly designated the backend to be used at any point. This will be defined in the main
entrypoint of our program, namely the `main` function defined in `src/main.rs`.

```rust , ignore
# mod data;
# mod model;
# mod training;
# 
use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{
    backend::{Autodiff, Wgpu},
#     data::dataset::Dataset,
    optim::AdamConfig,
};

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
}
```

In this example, we use the `Wgpu` backend which is compatible with any operating system and will
use the GPU. For other options, see the Burn README. This backend type takes the graphics API, the
float type and the int type as generic arguments that will be used during the training. The autodiff 
backend is simply the same backend, wrapped within the `Autodiff` struct which imparts differentiability \
to any backend.

We call the `train` function defined earlier with a directory for artifacts, the configuration of
the model (the number of digit classes is 10 and the hidden dimension is 512), the optimizer
configuration which in our case will be the default Adam configuration, and the device which can be
obtained from the backend.

You can now train your freshly created model with the command:

```console
cargo run --release
```

When running the example, you should see the training progression through a basic CLI dashboard:

<img title="a title" alt="Alt text" src="./training-output.png">
