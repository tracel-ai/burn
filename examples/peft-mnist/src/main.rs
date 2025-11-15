use burn::backend::{Autodiff, NdArray};
use peft_mnist::{benchmark, inference, training};

fn main() {
    type MyBackend = NdArray;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = Default::default();

    // Check command line arguments
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str());

    match mode {
        Some("infer") | Some("inference") => {
            inference::infer::<MyBackend>(device);
        }
        Some("bench") | Some("benchmark") => {
            benchmark::run_benchmarks();
        }
        Some("train") | None => {
            training::train::<MyAutodiffBackend>(device);
        }
        Some(other) => {
            eprintln!("Unknown mode: {}", other);
            eprintln!("Usage: peft-mnist [train|infer|bench]");
            std::process::exit(1);
        }
    }
}
