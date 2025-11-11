// Comprehensive benchmark for all PEFT variants: LoRA, DoRA, QLoRA, QDoRA
//
// Run with:
//   cargo bench --bench peft_comparison --features test-cuda
//   cargo bench --bench peft_comparison --features test-wgpu

use std::time::Instant;

// Backend selection based on features
#[cfg(feature = "test-cuda")]
use burn::backend::Autodiff;
#[cfg(feature = "test-cuda")]
use burn_cuda::{Cuda, CudaDevice};
#[cfg(feature = "test-cuda")]
type TestBackend = Autodiff<Cuda>;
#[cfg(feature = "test-cuda")]
fn get_device() -> CudaDevice {
    CudaDevice::default()
}

#[cfg(all(feature = "test-wgpu", not(feature = "test-cuda")))]
use burn::backend::Autodiff;
#[cfg(all(feature = "test-wgpu", not(feature = "test-cuda")))]
use burn_wgpu::{Wgpu, WgpuDevice};
#[cfg(all(feature = "test-wgpu", not(feature = "test-cuda")))]
type TestBackend = Autodiff<Wgpu>;
#[cfg(all(feature = "test-wgpu", not(feature = "test-cuda")))]
fn get_device() -> WgpuDevice {
    WgpuDevice::default()
}

#[cfg(not(any(feature = "test-cuda", feature = "test-wgpu")))]
compile_error!("This benchmark requires either 'test-cuda' or 'test-wgpu' feature enabled");

use burn::nn::LinearConfig;
use burn::{
    module::Module,
    tensor::{
        Distribution, Tensor,
        backend::Backend,
        quantization::{Calibration, QuantLevel, QuantScheme, QuantValue},
    },
};
use burn_peft::{DoRAConfig, LoRAConfig, QDoRAConfig, QLoRAConfig};

fn main() {
    println!("🔥 Burn PEFT Comprehensive Benchmark\n");
    println!("========================================\n");

    #[cfg(feature = "test-cuda")]
    println!("Backend: CUDA");
    #[cfg(all(feature = "test-wgpu", not(feature = "test-cuda")))]
    println!("Backend: WGPU");

    let device = get_device();

    // Test different model sizes
    let configs = vec![(512, 1024, "Medium", 8), (2048, 4096, "Large", 16)];

    for (d_in, d_out, name, rank) in configs {
        println!("\n📊 {} Model: [{} → {}], rank={}", name, d_in, d_out, rank);
        println!("─────────────────────────────────────────");

        benchmark_full_vs_peft(&device, d_in, d_out, rank);
        benchmark_lora_vs_dora(&device, d_in, d_out, rank);
        benchmark_merge_speedup(&device, d_in, d_out, rank);
        benchmark_quantized_variants(&device, d_in, d_out, rank);
        benchmark_memory_comparison(&device, d_in, d_out, rank);
    }
}

fn benchmark_full_vs_peft(
    device: &<TestBackend as Backend>::Device,
    d_in: usize,
    d_out: usize,
    rank: usize,
) {
    use burn_tensor::backend::AutodiffBackend;
    type InnerBackend = <TestBackend as AutodiffBackend>::InnerBackend;

    let batch_size = 32;
    let num_iters = 100;

    // Forward pass benchmarks (use inner backend to avoid autodiff overhead)
    let input_inner =
        Tensor::<InnerBackend, 2>::random([batch_size, d_in], Distribution::Default, device);

    let full_layer_inner = LinearConfig::new(d_in, d_out).init::<InnerBackend>(device);
    let lora_layer_inner = LoRAConfig::new(d_in, d_out)
        .with_rank(rank)
        .with_alpha((rank * 2) as f64)
        .init::<InnerBackend>(device);
    let dora_layer_inner = DoRAConfig::new(d_in, d_out)
        .with_rank(rank)
        .init::<InnerBackend>(device);

    // Warm-up
    let _ = full_layer_inner.forward(input_inner.clone());
    let _ = lora_layer_inner.forward(input_inner.clone());
    let _ = dora_layer_inner.forward(input_inner.clone());

    // Benchmark Full forward
    let start = Instant::now();
    for _ in 0..num_iters {
        let _ = full_layer_inner.forward(input_inner.clone());
    }
    let full_time = start.elapsed();
    let full_throughput = (num_iters * batch_size) as f64 / full_time.as_secs_f64();

    // Benchmark LoRA forward
    let start = Instant::now();
    for _ in 0..num_iters {
        let _ = lora_layer_inner.forward(input_inner.clone());
    }
    let lora_time = start.elapsed();
    let lora_throughput = (num_iters * batch_size) as f64 / lora_time.as_secs_f64();

    // Benchmark DoRA forward
    let start = Instant::now();
    for _ in 0..num_iters {
        let _ = dora_layer_inner.forward(input_inner.clone());
    }
    let dora_time = start.elapsed();
    let dora_throughput = (num_iters * batch_size) as f64 / dora_time.as_secs_f64();

    println!("\n  🏁 Full vs PEFT Forward Pass ({} iters):", num_iters);
    println!(
        "    Full:  {:?} ({:.0} samples/s)",
        full_time, full_throughput
    );
    println!(
        "    LoRA:  {:?} ({:.0} samples/s, {:.2}x vs Full)",
        lora_time,
        lora_throughput,
        lora_time.as_secs_f64() / full_time.as_secs_f64()
    );
    println!(
        "    DoRA:  {:?} ({:.0} samples/s, {:.2}x vs Full)",
        dora_time,
        dora_throughput,
        dora_time.as_secs_f64() / full_time.as_secs_f64()
    );

    // Backward pass comparison (use autodiff backend)
    let input_autodiff =
        Tensor::<TestBackend, 2>::random([batch_size, d_in], Distribution::Default, device);

    let full_layer_autodiff = LinearConfig::new(d_in, d_out).init::<TestBackend>(device);
    let lora_layer_autodiff = LoRAConfig::new(d_in, d_out)
        .with_rank(rank)
        .with_alpha((rank * 2) as f64)
        .init::<TestBackend>(device);

    let start = Instant::now();
    for _ in 0..num_iters {
        let output = full_layer_autodiff.forward(input_autodiff.clone());
        let loss = output.sum();
        let _ = loss.backward();
    }
    let full_backward = start.elapsed();
    let full_backward_throughput = (num_iters * batch_size) as f64 / full_backward.as_secs_f64();

    let start = Instant::now();
    for _ in 0..num_iters {
        let output = lora_layer_autodiff.forward(input_autodiff.clone());
        let loss = output.sum();
        let _ = loss.backward();
    }
    let lora_backward = start.elapsed();
    let lora_backward_throughput = (num_iters * batch_size) as f64 / lora_backward.as_secs_f64();

    println!("\n  🔄 Backward Pass ({} iters):", num_iters);
    println!(
        "    Full:  {:?} ({:.0} samples/s)",
        full_backward, full_backward_throughput
    );
    println!(
        "    LoRA:  {:?} ({:.0} samples/s, {:.2}x vs Full)",
        lora_backward,
        lora_backward_throughput,
        lora_backward.as_secs_f64() / full_backward.as_secs_f64()
    );
}

fn benchmark_merge_speedup(
    device: &<TestBackend as Backend>::Device,
    d_in: usize,
    d_out: usize,
    rank: usize,
) {
    use burn_tensor::backend::AutodiffBackend;
    type InnerBackend = <TestBackend as AutodiffBackend>::InnerBackend;

    let batch_size = 32;
    let num_iters = 100;
    let input =
        Tensor::<InnerBackend, 2>::random([batch_size, d_in], Distribution::Default, device);

    // LoRA layer (unmerged)
    let lora_config = LoRAConfig::new(d_in, d_out)
        .with_rank(rank)
        .with_alpha((rank * 2) as f64);
    let lora_unmerged = lora_config.init::<InnerBackend>(device);

    // LoRA layer (merged)
    let mut lora_merged = lora_unmerged.clone();
    lora_merged.merge_weights();

    // Warm-up
    let _ = lora_unmerged.forward(input.clone());
    let _ = lora_merged.forward(input.clone());

    // Benchmark unmerged
    let start = Instant::now();
    for _ in 0..num_iters {
        let _ = lora_unmerged.forward(input.clone());
    }
    let unmerged_time = start.elapsed();

    // Benchmark merged
    let start = Instant::now();
    for _ in 0..num_iters {
        let _ = lora_merged.forward(input.clone());
    }
    let merged_time = start.elapsed();

    println!("\n  🔀 LoRA Merge Speedup ({} iters):", num_iters);
    println!("    Unmerged: {:?}", unmerged_time);
    println!(
        "    Merged:   {:?} ({:.2}x speedup)",
        merged_time,
        unmerged_time.as_secs_f64() / merged_time.as_secs_f64()
    );
}

fn benchmark_lora_vs_dora(
    device: &<TestBackend as Backend>::Device,
    d_in: usize,
    d_out: usize,
    rank: usize,
) {
    use burn_tensor::backend::AutodiffBackend;
    type InnerBackend = <TestBackend as AutodiffBackend>::InnerBackend;

    let batch_size = 32;

    // Forward benchmarks (use inner backend)
    let input_inner =
        Tensor::<InnerBackend, 2>::random([batch_size, d_in], Distribution::Default, device);

    let lora_layer_inner = LoRAConfig::new(d_in, d_out)
        .with_rank(rank)
        .with_alpha((rank * 2) as f64)
        .init::<InnerBackend>(device);
    let dora_layer_inner = DoRAConfig::new(d_in, d_out)
        .with_rank(rank)
        .init::<InnerBackend>(device);

    // Warm-up
    let _ = lora_layer_inner.forward(input_inner.clone());
    let _ = dora_layer_inner.forward(input_inner.clone());

    // Benchmark LoRA forward
    let start = Instant::now();
    for _ in 0..100 {
        let _ = lora_layer_inner.forward(input_inner.clone());
    }
    let lora_time = start.elapsed();

    // Benchmark DoRA forward
    let start = Instant::now();
    for _ in 0..100 {
        let _ = dora_layer_inner.forward(input_inner.clone());
    }
    let dora_time = start.elapsed();

    println!("\n  ⚡ Forward Pass (100 iters):");
    println!("    LoRA:  {:?}", lora_time);
    println!(
        "    DoRA:  {:?} ({:.2}x vs LoRA)",
        dora_time,
        dora_time.as_secs_f64() / lora_time.as_secs_f64()
    );

    // Backward pass comparison (use autodiff backend)
    let lora_layer_autodiff = LoRAConfig::new(d_in, d_out)
        .with_rank(rank)
        .with_alpha((rank * 2) as f64)
        .init::<TestBackend>(device);
    let dora_layer_autodiff = DoRAConfig::new(d_in, d_out)
        .with_rank(rank)
        .init::<TestBackend>(device);

    // Benchmark LoRA backward (create fresh input each iteration to avoid non-leaf issues)
    let start = Instant::now();
    for _ in 0..100 {
        let input_autodiff =
            Tensor::<TestBackend, 2>::random([batch_size, d_in], Distribution::Default, device);
        let output = lora_layer_autodiff.forward(input_autodiff);
        let loss = output.sum();
        let _ = loss.backward();
    }
    let lora_backward_time = start.elapsed();

    // Benchmark DoRA backward (separate fresh inputs)
    let start = Instant::now();
    for _ in 0..100 {
        let input_autodiff =
            Tensor::<TestBackend, 2>::random([batch_size, d_in], Distribution::Default, device);
        let output = dora_layer_autodiff.forward(input_autodiff);
        let loss = output.sum();
        let _ = loss.backward();
    }
    let dora_backward_time = start.elapsed();

    println!("\n  🔄 Backward Pass (100 iters):");
    println!("    LoRA:  {:?}", lora_backward_time);
    println!(
        "    DoRA:  {:?} ({:.2}x vs LoRA)",
        dora_backward_time,
        dora_backward_time.as_secs_f64() / lora_backward_time.as_secs_f64()
    );
}

fn benchmark_quantized_variants(
    device: &<TestBackend as Backend>::Device,
    d_in: usize,
    d_out: usize,
    rank: usize,
) {
    use burn_tensor::backend::AutodiffBackend;
    type InnerBackend = <TestBackend as AutodiffBackend>::InnerBackend;

    let batch_size = 32;
    let base_weight =
        Tensor::<InnerBackend, 2>::random([d_in, d_out], Distribution::Default, device);

    // Quantization scheme (4-bit)
    let quant_scheme_4bit = QuantScheme::default()
        .with_value(QuantValue::Q4F)
        .with_level(QuantLevel::Tensor);

    // Quantize the base weight
    let range = burn::tensor::quantization::compute_range(
        &quant_scheme_4bit,
        &base_weight,
        &Calibration::MinMax,
    );
    let qparams = burn::tensor::quantization::compute_q_params(&quant_scheme_4bit, range);
    let base_weight_quantized = base_weight.clone().quantize(&quant_scheme_4bit, qparams);

    // Standard LoRA
    let lora_config = LoRAConfig::new(d_in, d_out)
        .with_rank(rank)
        .with_alpha((rank * 2) as f64);
    let lora_layer =
        lora_config.init_with_base_weight::<InnerBackend>(base_weight.clone(), None, device);

    // QLoRA (4-bit quantization)
    let qlora_config = QLoRAConfig::new(d_in, d_out)
        .with_rank(rank)
        .with_alpha((rank * 2) as f64);
    let qlora_layer = qlora_config.init_with_quantized_weight::<InnerBackend>(
        base_weight_quantized.clone(),
        None,
        device,
    );

    // Standard DoRA
    let dora_config = DoRAConfig::new(d_in, d_out).with_rank(rank);
    let dora_layer =
        dora_config.init_with_base_weight::<InnerBackend>(base_weight.clone(), None, device);

    // QDoRA (4-bit quantization)
    let qdora_config = QDoRAConfig::new(d_in, d_out).with_rank(rank);
    let qdora_layer = qdora_config.init_with_quantized_weight::<InnerBackend>(
        base_weight_quantized,
        None,
        device,
    );

    let input =
        Tensor::<InnerBackend, 2>::random([batch_size, d_in], Distribution::Default, device);

    // Warm-up
    let _ = lora_layer.forward(input.clone());
    let _ = qlora_layer.forward(input.clone());
    let _ = dora_layer.forward(input.clone());
    let _ = qdora_layer.forward(input.clone());

    // Benchmark QLoRA vs LoRA
    let start = Instant::now();
    for _ in 0..100 {
        let _ = lora_layer.forward(input.clone());
    }
    let lora_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..100 {
        let _ = qlora_layer.forward(input.clone());
    }
    let qlora_time = start.elapsed();

    // Benchmark QDoRA vs DoRA
    let start = Instant::now();
    for _ in 0..100 {
        let _ = dora_layer.forward(input.clone());
    }
    let dora_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..100 {
        let _ = qdora_layer.forward(input.clone());
    }
    let qdora_time = start.elapsed();

    println!("\n  💾 Quantized Forward Pass (100 iters):");
    println!("    LoRA:   {:?}", lora_time);
    println!(
        "    QLoRA:  {:?} ({:.2}x vs LoRA)",
        qlora_time,
        qlora_time.as_secs_f64() / lora_time.as_secs_f64()
    );
    println!("    DoRA:   {:?}", dora_time);
    println!(
        "    QDoRA:  {:?} ({:.2}x vs DoRA)",
        qdora_time,
        qdora_time.as_secs_f64() / dora_time.as_secs_f64()
    );
}

fn benchmark_memory_comparison(
    device: &<TestBackend as Backend>::Device,
    d_in: usize,
    d_out: usize,
    rank: usize,
) {
    use burn_tensor::backend::AutodiffBackend;
    type InnerBackend = <TestBackend as AutodiffBackend>::InnerBackend;

    let base_weight =
        Tensor::<InnerBackend, 2>::random([d_in, d_out], Distribution::Default, device);

    // Quantization scheme (4-bit)
    let quant_scheme_4bit = QuantScheme::default()
        .with_value(QuantValue::Q4F)
        .with_level(QuantLevel::Tensor);

    let range = burn::tensor::quantization::compute_range(
        &quant_scheme_4bit,
        &base_weight,
        &Calibration::MinMax,
    );
    let qparams = burn::tensor::quantization::compute_q_params(&quant_scheme_4bit, range);
    let base_weight_quantized = base_weight.clone().quantize(&quant_scheme_4bit, qparams);

    // LoRA
    let lora_config = LoRAConfig::new(d_in, d_out)
        .with_rank(rank)
        .with_alpha((rank * 2) as f64);
    let lora_layer =
        lora_config.init_with_base_weight::<InnerBackend>(base_weight.clone(), None, device);

    // DoRA
    let dora_config = DoRAConfig::new(d_in, d_out).with_rank(rank);
    let dora_layer =
        dora_config.init_with_base_weight::<InnerBackend>(base_weight.clone(), None, device);

    // QLoRA (4-bit)
    let qlora_config = QLoRAConfig::new(d_in, d_out)
        .with_rank(rank)
        .with_alpha((rank * 2) as f64);
    let qlora_layer = qlora_config.init_with_quantized_weight::<InnerBackend>(
        base_weight_quantized.clone(),
        None,
        device,
    );

    // QDoRA (4-bit)
    let qdora_config = QDoRAConfig::new(d_in, d_out).with_rank(rank);
    let qdora_layer = qdora_config.init_with_quantized_weight::<InnerBackend>(
        base_weight_quantized,
        None,
        device,
    );

    let lora_params = lora_layer.num_params();
    let dora_params = dora_layer.num_params();
    let qlora_params = qlora_layer.num_params();
    let qdora_params = qdora_layer.num_params();

    let base_params = d_in * d_out;
    let trainable_lora = rank * (d_in + d_out);
    let trainable_dora = trainable_lora + d_out; // +magnitude vector

    println!("\n  📊 Memory Footprint:");
    println!("    Base params:     {}", base_params);
    println!(
        "    LoRA total:      {} ({:.1}% trainable)",
        lora_params,
        100.0 * trainable_lora as f64 / lora_params as f64
    );
    println!(
        "    DoRA total:      {} ({:.1}% trainable)",
        dora_params,
        100.0 * trainable_dora as f64 / dora_params as f64
    );
    println!(
        "    QLoRA total:     {} (4-bit base, {:.1}% trainable)",
        qlora_params,
        100.0 * trainable_lora as f64 / qlora_params as f64
    );
    println!(
        "    QDoRA total:     {} (4-bit base, {:.1}% trainable)",
        qdora_params,
        100.0 * trainable_dora as f64 / qdora_params as f64
    );

    // Estimate memory savings (FP32 baseline)
    let fp32_size = base_params * 4; // 4 bytes per FP32

    // LoRA: FP32 base + FP32 adapters
    let lora_total = (base_params + trainable_lora) * 4;

    // DoRA: FP32 base + FP32 adapters + FP32 magnitude
    let dora_total = (base_params + trainable_dora) * 4;

    // QLoRA: 4-bit base + FP32 adapters
    let qlora_base_size = base_params / 2; // 4-bit = 0.5 bytes
    let qlora_adapter_size = trainable_lora * 4; // Adapters are FP32
    let qlora_total = qlora_base_size + qlora_adapter_size;

    // QDoRA: 4-bit base + FP32 adapters + FP32 magnitude
    let qdora_total = qlora_base_size + trainable_dora * 4;

    println!("\n  💰 Memory Savings (vs FP32 full fine-tuning):");
    println!(
        "    LoRA:   {:.1}% trainable params ({:.1} MB total)",
        100.0 * (1.0 - (trainable_lora * 4) as f64 / fp32_size as f64),
        lora_total as f64 / 1_000_000.0
    );
    println!(
        "    DoRA:   {:.1}% trainable params ({:.1} MB total)",
        100.0 * (1.0 - (trainable_dora * 4) as f64 / fp32_size as f64),
        dora_total as f64 / 1_000_000.0
    );
    println!(
        "    QLoRA:  {:.1}% total savings ({:.1} MB → {:.1} MB)",
        100.0 * (1.0 - qlora_total as f64 / fp32_size as f64),
        fp32_size as f64 / 1_000_000.0,
        qlora_total as f64 / 1_000_000.0
    );
    println!(
        "    QDoRA:  {:.1}% total savings ({:.1} MB → {:.1} MB)",
        100.0 * (1.0 - qdora_total as f64 / fp32_size as f64),
        fp32_size as f64 / 1_000_000.0,
        qdora_total as f64 / 1_000_000.0
    );
}
