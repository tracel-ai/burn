use crate::model::{LoRAMLP, SimpleMLP};
use burn::{
    backend::{Autodiff, NdArray},
    module::Module,
    tensor::{Distribution, Tensor, backend::Backend},
};
use std::time::Instant;

type BenchBackend = NdArray;
type BenchAutodiffBackend = Autodiff<NdArray>;

pub fn run_benchmarks() {
    let device = Default::default();

    println!("🔥 Burn PEFT Benchmark: LoRA vs Full Fine-Tuning");
    println!("================================================\n");

    // Small model
    println!("📊 Small Model: 784 → 512 → 256 → 10");
    benchmark_mnist_model(&device);

    // Medium model
    println!("\n📊 Medium Model: 784 → 2048 → 1024 → 10");
    benchmark_medium_model(&device);
}

fn benchmark_mnist_model(device: &<BenchBackend as Backend>::Device) {
    let batch_size = 64;
    let input = Tensor::<BenchBackend, 2>::random([batch_size, 784], Distribution::Default, device);

    // Full MLP
    let full_model = SimpleMLP::<BenchBackend>::new(device);
    let full_params = full_model.num_params();

    // LoRA MLP
    let lora_model = LoRAMLP::from_pretrained(
        SimpleMLP::<BenchBackend>::new(device),
        8,    // rank
        16.0, // alpha
        device,
    );
    let lora_params = lora_model.num_params();

    // Calculate trainable parameters for LoRA
    // Base weights are frozen, only LoRA adapters are trainable
    let rank = 8;
    let lora_trainable = (512 * rank + rank * 784)  // fc1: A + B
                       + (256 * rank + rank * 512)  // fc2: A + B
                       + (10 * rank + rank * 256); // fc3: A + B

    println!("\n  Parameters:");
    println!("    Full Model:        {:>10}", full_params);
    println!("    LoRA Total:        {:>10}", lora_params);
    println!(
        "    LoRA Trainable:    {:>10}  ({:.1}% of full)",
        lora_trainable,
        100.0 * lora_trainable as f64 / full_params as f64
    );
    println!(
        "    LoRA Frozen:       {:>10}  ({:.1}% of full)",
        lora_params - lora_trainable,
        100.0 * (lora_params - lora_trainable) as f64 / full_params as f64
    );

    println!("\n  Memory (FP32):");
    let full_mb = (full_params * 4) as f64 / 1_048_576.0;
    let lora_total_mb = (lora_params * 4) as f64 / 1_048_576.0;
    let lora_trainable_mb = (lora_trainable * 4) as f64 / 1_048_576.0;
    println!("    Full Model:        {:>8.2} MB", full_mb);
    println!("    LoRA Total:        {:>8.2} MB", lora_total_mb);
    println!(
        "    LoRA Trainable:    {:>8.2} MB  (only these need gradients)",
        lora_trainable_mb
    );
    println!("    Gradient Memory:");
    println!("      Full Training:   {:>8.2} MB  (all params)", full_mb);
    println!(
        "      LoRA Training:   {:>8.2} MB  (adapters only)",
        lora_trainable_mb
    );
    println!(
        "      Savings:         {:>8.2} MB  ({:.1}%)",
        full_mb - lora_trainable_mb,
        100.0 * (full_mb - lora_trainable_mb) / full_mb
    );

    // Forward pass benchmark
    println!("\n  Forward Pass (1000 iterations):");

    // Warm-up
    let _ = full_model.forward(input.clone());
    let _ = lora_model.forward(input.clone());

    // Benchmark full
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = full_model.forward(input.clone());
    }
    let full_time = start.elapsed();

    // Benchmark LoRA (unmerged)
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = lora_model.forward(input.clone());
    }
    let lora_time = start.elapsed();

    // Benchmark LoRA (merged)
    let mut lora_merged = lora_model.clone();
    lora_merged.merge_weights();

    let start = Instant::now();
    for _ in 0..1000 {
        let _ = lora_merged.forward(input.clone());
    }
    let lora_merged_time = start.elapsed();

    println!("    Full:              {:?}", full_time);
    println!(
        "    LoRA (unmerged):   {:?}  ({:.2}x)",
        lora_time,
        lora_time.as_secs_f64() / full_time.as_secs_f64()
    );
    println!(
        "    LoRA (merged):     {:?}  ({:.2}x)",
        lora_merged_time,
        lora_merged_time.as_secs_f64() / full_time.as_secs_f64()
    );

    // Backward pass benchmark (needs autodiff)
    println!("\n  Backward Pass (100 iterations):");

    // Create autodiff models for gradient computation
    let full_model_ad = SimpleMLP::<BenchAutodiffBackend>::new(device);
    let lora_model_ad = LoRAMLP::from_pretrained(
        SimpleMLP::<BenchAutodiffBackend>::new(device),
        8,
        16.0,
        device,
    );
    let input_ad =
        Tensor::<BenchAutodiffBackend, 2>::random([batch_size, 784], Distribution::Default, device);

    let start = Instant::now();
    for _ in 0..100 {
        let output = full_model_ad.forward(input_ad.clone());
        let loss = output.sum();
        let _ = loss.backward();
    }
    let full_backward_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..100 {
        let output = lora_model_ad.forward(input_ad.clone());
        let loss = output.sum();
        let _ = loss.backward();
    }
    let lora_backward_time = start.elapsed();

    println!("    Full:              {:?}", full_backward_time);
    println!(
        "    LoRA:              {:?}  ({:.2}x)",
        lora_backward_time,
        lora_backward_time.as_secs_f64() / full_backward_time.as_secs_f64()
    );
}

fn benchmark_medium_model(device: &<BenchBackend as Backend>::Device) {
    use burn::nn::{Linear, LinearConfig, Relu};

    #[derive(Module, Debug)]
    struct MediumMLP<B: Backend> {
        fc1: Linear<B>,
        fc2: Linear<B>,
        fc3: Linear<B>,
        activation: Relu,
    }

    impl<B: Backend> MediumMLP<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                fc1: LinearConfig::new(784, 2048).init(device),
                fc2: LinearConfig::new(2048, 1024).init(device),
                fc3: LinearConfig::new(1024, 10).init(device),
                activation: Relu::new(),
            }
        }

        fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let x = input.flatten(1, 1);
            let x = self.fc1.forward(x);
            let x = self.activation.forward(x);
            let x = self.fc2.forward(x);
            let x = self.activation.forward(x);
            self.fc3.forward(x)
        }
    }

    let batch_size = 64;
    let input = Tensor::<BenchBackend, 2>::random([batch_size, 784], Distribution::Default, device);

    let full_model = MediumMLP::<BenchBackend>::new(device);
    let full_params = full_model.num_params();

    // Calculate LoRA trainable params for medium model
    let rank = 8;
    let lora_trainable = (2048 * rank + rank * 784)   // fc1: A + B
                       + (1024 * rank + rank * 2048)  // fc2: A + B
                       + (10 * rank + rank * 1024); // fc3: A + B

    println!("\n  Parameters:");
    println!("    Full Model:        {:>10}", full_params);
    println!(
        "    LoRA Trainable:    {:>10}  ({:.1}% of full)",
        lora_trainable,
        100.0 * lora_trainable as f64 / full_params as f64
    );

    println!("\n  Memory (FP32):");
    let full_mb = (full_params * 4) as f64 / 1_048_576.0;
    let lora_trainable_mb = (lora_trainable * 4) as f64 / 1_048_576.0;
    println!("    Full Model:        {:>8.2} MB", full_mb);
    println!("    LoRA Trainable:    {:>8.2} MB", lora_trainable_mb);
    println!(
        "    Gradient Savings:  {:>8.2} MB  ({:.1}%)",
        full_mb - lora_trainable_mb,
        100.0 * (full_mb - lora_trainable_mb) / full_mb
    );

    println!("\n  QLoRA (4-bit) Memory:");
    let qlora_base_mb = (full_params as f64 * 0.5) / 1_048_576.0; // 4-bit = 0.5 bytes
    let qlora_total_mb = qlora_base_mb + lora_trainable_mb;
    println!(
        "    Base (4-bit):      {:>8.2} MB  (quantized)",
        qlora_base_mb
    );
    println!("    Adapters (FP32):   {:>8.2} MB", lora_trainable_mb);
    println!("    Total:             {:>8.2} MB", qlora_total_mb);
    println!(
        "    vs Full Model:     {:>8.2} MB saved ({:.1}%)",
        full_mb - qlora_total_mb,
        100.0 * (full_mb - qlora_total_mb) / full_mb
    );

    // Forward pass benchmark
    println!("\n  Forward Pass (100 iterations):");

    let start = Instant::now();
    for _ in 0..100 {
        let _ = full_model.forward(input.clone());
    }
    let full_time = start.elapsed();

    println!("    Full:              {:?}", full_time);
}
