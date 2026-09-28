//! What a program sees when the device fails under it.
//!
//! ```sh
//! # Inference: batch 5 fails; the loop reports it, retries it, and finishes.
//! cargo run -p mnist --example mnist-fault --release --features cuda,fault-injection -- inference refused 5
//! # Inference: batch 5 poisons the device; the loop reports it and stops.
//! cargo run -p mnist --example mnist-fault --release --features cuda,fault-injection -- inference poisoned 5
//! # Training: step 20 fails; training stops with the failure as the cause.
//! cargo run -p mnist --example mnist-fault --release --features cuda,fault-injection -- training refused 20
//! # The same on Vulkan or wgpu. Neither can fault on an out-of-bounds write —
//! # WebGPU clamps every access — so `poisoned` only happens there after a real driver loss.
//! cargo run -p mnist --example mnist-fault --release --features vulkan,fault-injection -- inference refused 5
//! ```

use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, transform::Mapper, vision::MnistDataset},
    },
    tensor::{Device, ExecutionError},
};
use mnist::{
    data::{MnistBatcher, MnistMapper},
    fault::{self, Fault},
    model::Model,
    training,
};

const BATCH_SIZE: usize = 256;
const BATCHES: usize = 10;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [mode, fault, step] = args.as_slice() else {
        eprintln!("usage: mnist-fault <inference|training> <refused|poisoned> <step>");
        std::process::exit(2);
    };
    let fault: Fault = fault.parse().unwrap_or_else(|err| panic!("{err}"));
    let step: usize = step.parse().expect("the step is a number");

    let device = select_device();

    match mode.as_str() {
        "inference" => inference(device, fault, step),
        "training" => {
            fault::schedule(fault, step);
            training::run(device);
        }
        other => panic!("unknown mode `{other}`, expected `inference` or `training`"),
    }
}

#[allow(unreachable_code)]
fn select_device() -> Device {
    #[cfg(feature = "cuda")]
    return Device::cuda(burn::tensor::DeviceIndex::Default);
    #[cfg(feature = "rocm")]
    return Device::rocm(burn::tensor::DeviceIndex::Default);
    #[cfg(feature = "vulkan")]
    return Device::vulkan(burn::tensor::DeviceKind::DefaultDevice);
    #[cfg(feature = "metal")]
    return Device::metal(burn::tensor::DeviceKind::DefaultDevice);
    #[cfg(feature = "wgpu")]
    return Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);

    panic!("enable one of the `cuda`, `rocm`, `vulkan`, `metal` or `wgpu` features")
}

/// Classify a few batches, handling each batch's result where it is read.
fn inference(device: Device, fault: Fault, at: usize) {
    let model = Model::new(&device);
    let dataset = MnistDataset::test();
    let mapper = MnistMapper::default();
    let batcher = MnistBatcher::default();

    for index in 0..BATCHES {
        let items = (index * BATCH_SIZE..(index + 1) * BATCH_SIZE)
            .filter_map(|i| dataset.get(i).ok())
            .map(|item| mapper.map(&item))
            .collect::<Vec<_>>();
        let batch = batcher.batch(items, &device);

        let inject = (index == at).then_some(fault);
        match classify(&model, batch.clone(), inject) {
            Ok(correct) => println!("batch {index}: {correct}/{BATCH_SIZE} correct"),
            Err(err) if err.is_device_poisoned() => {
                eprintln!("batch {index}: the device is poisoned, stopping.\n{err}");
                std::process::exit(1);
            }
            Err(err) => {
                eprintln!("batch {index}: failed, retrying it.\n{err}");
                let correct =
                    classify(&model, batch, None).expect("the retry runs on a healthy device");
                println!("batch {index} (retried): {correct}/{BATCH_SIZE} correct");
            }
        }
    }
    println!("done: every batch was classified");
}

/// How many of the batch's images the model gets right.
///
/// The one sync point is the read at the end: an injected fault, wherever it
/// sits in the computation, comes back from here.
fn classify(
    model: &Model,
    batch: mnist::data::MnistBatch,
    fault: Option<Fault>,
) -> Result<i64, ExecutionError> {
    let logits = model.forward(batch.images);
    let logits = match fault {
        Some(fault) => fault::inject_float(logits, fault),
        None => logits,
    };
    let predictions = logits.argmax(1).flatten::<1>(0, 1);
    let correct = predictions.equal(batch.targets).int().sum();

    let data = correct.try_into_data()?;
    Ok(data.iter::<i64>().next().unwrap_or_default())
}
