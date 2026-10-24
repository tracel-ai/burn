//! Measure where the AG News classifier's inference spends its device time, and print the
//! report.
//!
//! ```sh
//! cargo run -p text-classification --example profile --release --features cuda,profile
//! cargo run -p text-classification --example profile --release --features cuda,profile -- now.json
//! # In half precision: `f16`, `bf16` or `flex32`, as the other examples take it.
//! cargo run -p text-classification --example profile --release --features cuda,profile,f16
//! ```
//!
//! The model is the one `ag-news-train` trains, at random weights: what a pass costs depends on
//! its shapes, not its values. With a path, the measurement is also saved there, for
//! `burn-instrument-report` to hold a later run against.

#[cfg(any(
    feature = "wgpu",
    feature = "vulkan",
    feature = "metal",
    feature = "cuda",
    feature = "rocm"
))]
mod profile {
    use burn::instrument::Instrument;
    use burn::nn::{attention::SeqLengthOption, transformer::TransformerEncoderConfig};
    use burn::tensor::{Device, DeviceConfig, Distribution, Element, Int, Tensor};
    use burn_instrument_report::Report;
    use text_classification::data::TextClassificationInferenceBatch;
    use text_classification::model::TextClassificationModelConfig;

    #[cfg(not(any(feature = "f16", feature = "bf16", feature = "flex32")))]
    type ElemType = f32;
    #[cfg(feature = "f16")]
    type ElemType = burn::tensor::f16;
    #[cfg(feature = "bf16")]
    type ElemType = burn::tensor::bf16;
    #[cfg(feature = "flex32")]
    type ElemType = burn::tensor::flex32;

    /// The vocabulary of `bert-base-cased`, the tokenizer the model is trained with, stated here
    /// so a measurement does not download it.
    const VOCAB_SIZE: usize = 28996;
    /// The AG News categories.
    const CLASSES: usize = 4;
    /// The batch and sequence length `ag-news-train` trains at.
    const BATCH_SIZE: usize = 32;
    const SEQ_LENGTH: usize = 256;

    pub fn run(mut device: Device) {
        // The model computes in this dtype, and the arithmetic peak it is scored against is
        // probed at it.
        device
            .configure(DeviceConfig::default().float_dtype(ElemType::dtype()))
            .unwrap();

        let model = TextClassificationModelConfig::new(
            TransformerEncoderConfig::new(128, 512, 4, 4)
                .with_norm_first(true)
                .with_quiet_softmax(true),
            CLASSES,
            VOCAB_SIZE,
            SeqLengthOption::Fixed(SEQ_LENGTH),
        )
        .init(&device);
        let tokens = Tensor::<2, Int>::random(
            [BATCH_SIZE, SEQ_LENGTH],
            Distribution::Uniform(0.0, VOCAB_SIZE as f64),
            &device,
        );
        // No padding: every position is a token.
        let mask_pad = Tensor::<2, Int>::zeros([BATCH_SIZE, SEQ_LENGTH], &device).equal_elem(1);
        let batch = TextClassificationInferenceBatch::new(tokens, mask_pad);

        let measurement = Instrument::new(&device).measure(|| model.infer(batch.clone()));
        println!("{}", Report::new(&measurement));

        if let Some(path) = std::env::args().nth(1) {
            let file = std::fs::File::create(&path).expect("the measurement file can be created");
            measurement.write(file).expect("the measurement is written");
            println!("saved {path}");
        }
    }
}

fn main() {
    #[cfg(feature = "wgpu")]
    profile::run(burn::tensor::Device::wgpu(
        burn::tensor::DeviceKind::DefaultDevice,
    ));
    #[cfg(feature = "vulkan")]
    profile::run(burn::tensor::Device::vulkan(
        burn::tensor::DeviceKind::DefaultDevice,
    ));
    #[cfg(feature = "metal")]
    profile::run(burn::tensor::Device::metal(
        burn::tensor::DeviceKind::DefaultDevice,
    ));
    #[cfg(feature = "cuda")]
    profile::run(burn::tensor::Device::cuda(
        burn::tensor::DeviceIndex::Default,
    ));
    #[cfg(feature = "rocm")]
    profile::run(burn::tensor::Device::rocm(
        burn::tensor::DeviceIndex::Default,
    ));
    #[cfg(not(any(
        feature = "wgpu",
        feature = "vulkan",
        feature = "metal",
        feature = "cuda",
        feature = "rocm"
    )))]
    eprintln!(
        "profiling reads a CubeCL runtime's kernel launches: enable one of the wgpu, vulkan, \
         metal, cuda or rocm features"
    );
}
