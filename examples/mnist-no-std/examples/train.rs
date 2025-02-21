use clap::Parser;
use mnist_no_std::{proto::*, train::no_std_world};
use rand::{seq::SliceRandom, Rng};

/// Trains a new model and exports it to the given path.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short, long, default_value_t = 6)]
    num_epochs: usize,
    #[arg(short, long, default_value_t = 64)]
    batch_size: usize,
    #[arg(short, long, default_value_t = 0.0001)]
    learning_rate: f64,
    #[arg(short, long, default_value = "model.bin")]
    output: String,
}

fn convert_datasets(images: &[u8], labels: &[u8]) -> Vec<(MnistImage, u8)> {
    let mut datasets: Vec<(MnistImage, u8)> = images
        .chunks_exact(MNIST_IMAGE_SIZE)
        .map(|v| v.try_into().unwrap())
        .zip(labels.iter().copied())
        .collect();
    datasets.shuffle(&mut rand::rng());
    datasets
}

fn main() {
    let args = Args::parse();
    // Download mnist data, keep the same URL with burn.
    // Originally copy from burn/crates/burn-dataset/src/vision/mnist.rs
    const BASE_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist/";
    let data = mnist::MnistBuilder::new()
        .base_url(BASE_URL)
        .base_path(
            std::env::temp_dir()
                .join("example_mnist_no_std/")
                .as_path()
                .to_str()
                .unwrap(),
        )
        .download_and_extract()
        .training_set_length(60_000)
        .validation_set_length(10_000)
        .test_set_length(0)
        .finalize();
    // Initialize trainer
    let seed: u64 = rand::rng().random();
    no_std_world::initialize(seed, args.learning_rate);
    // Prepare datasets
    let train_datasets = convert_datasets(&data.trn_img, &data.trn_lbl);
    let valid_datasets = convert_datasets(&data.val_img, &data.val_lbl);
    // Training loop, Originally inspired by burn/crates/custom-training-loop
    //
    // Normally there is no println in no_std, the caller must invoke functions
    // step by step and receive feedback from no_std_world. For example, in
    // TrustZone, there are two systems running on the same machine: one is
    // Linux, and the other is TEEOS (the no_std world, bare metal env). The
    // caller from Linux invokes functions via SMC (Secure Monitor Call)
    // repeatedly, receives output through shared memory, and prints it to the
    // screen.
    for epoch in 1..args.num_epochs + 1 {
        for (iteration, data) in train_datasets.chunks(args.batch_size).enumerate() {
            let images: Vec<MnistImage> = data.iter().map(|v| v.0).collect();
            let labels: Vec<u8> = data.iter().map(|v| v.1).collect();
            let output = no_std_world::train(bytemuck::cast_slice(images.as_slice()), &labels);
            println!(
                "[Train - Epoch {} - Iteration {}] Loss {:.3} | Accuracy {:.3} %",
                epoch, iteration, output.loss, output.accuracy,
            );
        }

        for (iteration, data) in valid_datasets.chunks(args.batch_size).enumerate() {
            let images: Vec<MnistImage> = data.iter().map(|v| v.0).collect();
            let labels: Vec<u8> = data.iter().map(|v| v.1).collect();
            let output = no_std_world::valid(bytemuck::cast_slice(images.as_slice()), &labels);
            println!(
                "[Valid - Epoch {} - Iteration {}] Loss {:.3} | Accuracy {:.3} %",
                epoch, iteration, output.loss, output.accuracy,
            );
        }
    }
    // Export the model to the given path
    let record = no_std_world::export();
    let output_path = std::path::absolute(&args.output).unwrap();
    println!("Export record to \"{}\"", output_path.display());
    std::fs::write(&output_path, &record).unwrap();
}
