#![recursion_limit = "131"]
use burn::{data::dataset::Dataset, prelude::*};
use guide::inference;

fn main() {
    let device = Device::wgpu(DeviceKind::DefaultDevice);

    // All the training artifacts are saved in this directory
    let artifact_dir = "/tmp/guide";

    // Infer the model
    inference::infer(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(42)
            .unwrap(),
    );
}
