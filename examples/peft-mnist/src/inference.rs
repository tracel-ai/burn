use crate::{
    model::{LoRAMLP, SimpleMLP},
    training::TrainingConfig,
};
use burn::{
    data::dataset::{
        Dataset,
        vision::{MnistDataset, MnistItem},
    },
    prelude::*,
    record::{CompactRecorder, Recorder},
};

const ARTIFACT_DIR: &str = "./tmp/peft-mnist";

pub fn infer<B: Backend>(device: B::Device) {
    // Load configuration
    let config = TrainingConfig::load(format!("{ARTIFACT_DIR}/config.json").as_str())
        .expect("Config should exist; run train first");

    println!("🔥 Burn PEFT Inference");
    println!("========================================");
    println!(
        "Loading trained LoRA model (rank={}, alpha={})...\n",
        config.lora_rank, config.lora_alpha
    );

    // Initialize a baseline model
    let baseline = SimpleMLP::<B>::new(&device);

    // Convert to LoRA with same config as training
    let model = LoRAMLP::from_pretrained(baseline, config.lora_rank, config.lora_alpha, &device);

    // Load the trained weights
    let record = CompactRecorder::new()
        .load(format!("{ARTIFACT_DIR}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = model.load_record(record);

    // Get test dataset
    let test_dataset = MnistDataset::test();

    println!(
        "Running inference on {} test samples...\n",
        test_dataset.len()
    );

    // Run inference on first 10 samples
    let num_samples = 10.min(test_dataset.len());
    let mut correct = 0;

    for i in 0..num_samples {
        let item = test_dataset.get(i).unwrap();
        let prediction = run_inference(&model, item.clone(), &device);

        let actual = item.label;
        let is_correct = prediction == actual;
        if is_correct {
            correct += 1;
        }

        println!(
            "Sample {}: Predicted: {}, Actual: {}, {}",
            i + 1,
            prediction,
            actual,
            if is_correct { "✓" } else { "✗" }
        );
    }

    println!(
        "\nAccuracy on {} samples: {}/{} ({:.1}%)",
        num_samples,
        correct,
        num_samples,
        100.0 * correct as f32 / num_samples as f32
    );
}

fn run_inference<B: Backend>(model: &LoRAMLP<B>, item: MnistItem, device: &B::Device) -> u8 {
    // Prepare the image
    let data = TensorData::from(item.image);
    let tensor = Tensor::<B, 2>::from_data(data.convert::<f32>(), device);

    // Normalize the same way as training
    let tensor = ((tensor / 255.0) - 0.1307) / 0.3081;
    let tensor = tensor.reshape([1, 1, 28, 28]);

    // Flatten for MLP input
    let tensor = tensor.reshape([1, 784]);

    // Run forward pass
    let output = model.forward(tensor);

    // Get prediction (argmax)
    let prediction = output.argmax(1);
    let prediction_data = prediction.into_data();

    prediction_data.to_vec::<i64>().unwrap()[0] as u8
}
