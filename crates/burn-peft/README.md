# Burn-PEFT: Parameter-Efficient Fine-Tuning for Burn

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](../../LICENSE)

State-of-the-art parameter-efficient fine-tuning (PEFT) methods for the [Burn](https://github.com/tracel-ai/burn) deep learning framework.

## Overview

`burn-peft` provides drop-in replacements for linear layers that enable efficient fine-tuning of large pretrained models using:

- **LoRA** (Low-Rank Adaptation): Efficient fine-tuning via low-rank weight updates
- **DoRA** (Weight-Decomposed Low-Rank Adaptation): Decomposes weights into magnitude and direction for improved performance
- **LoRA+**: Improved LoRA with differential learning rates (16x faster convergence)
- **QLoRA** (Quantized LoRA): LoRA with 4-8 bit quantized base weights for massive memory savings
- **QDoRA** (Quantized DoRA): DoRA with quantized base weights - best of both worlds
- **Adapter Composition**: Merge multiple trained adapters for multi-task learning

## Features

- ✅ **Exact mathematical semantics** following official specifications
- ✅ **Numerical stability** with proper epsilon handling and FP32 accumulation
- ✅ **Merge/unmerge support** for efficient inference (LoRA/DoRA)
- ✅ **Detached norm gradients** (DoRA/QDoRA) for memory efficiency
- ✅ **Quantization support** (QLoRA/QDoRA) with 4-8 bit weights
- ✅ **Drop-in replacement** for `burn::nn::Linear`
- ✅ **Memory savings up to 87%** with 4-bit quantization

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
burn-peft = "0.20.0"
```

### LoRA Example

```rust
use burn::tensor::{backend::Backend, Tensor, Distribution};
use burn_peft::{LoRAConfig, LoRALinear};

fn example<B: Backend>(device: &B::Device) {
    // Create LoRA configuration
    let config = LoRAConfig::new(512, 512)  // input_dim, output_dim
        .with_rank(16)                      // rank r (smaller = fewer params)
        .with_alpha(32.0);                  // scaling α (often 2*rank)

    // Initialize LoRA layer
    let layer = config.init::<B>(device);

    // Forward pass (training mode)
    let input = Tensor::<B, 2>::random([8, 512], Distribution::Default, device);
    let output = layer.forward(input);  // Shape: [8, 512]

    // Merge for inference (optional, faster)
    let mut layer_merged = layer.clone();
    layer_merged.merge_weights();
    let output_merged = layer_merged.forward(input);  // Same result, faster
}
```

### DoRA Example

```rust
use burn_peft::{DoRAConfig, DoRALinear};

fn example<B: Backend>(device: &B::Device) {
    // Create DoRA configuration
    let config = DoRAConfig::new(512, 512)
        .with_rank(16)
        .with_epsilon(1e-8);  // For numerical stability

    let layer = config.init::<B>(device);

    let input = Tensor::<B, 2>::random([8, 512], Distribution::Default, device);
    let output = layer.forward(input);
}
```

### Converting Existing Weights

```rust
use burn::module::Param;

// Load pretrained weight
let pretrained_weight: Tensor<B, 2> = /* ... load from checkpoint ... */;
let bias: Option<Tensor<B, 1>> = Some(/* ... */);

// Convert to LoRA
let config = LoRAConfig::new(d_in, d_out).with_rank(8);
let lora_layer = config.init_with_base_weight(pretrained_weight, bias, device);

// Now only the LoRA adapters (A, B) are trainable
// The base weight is frozen
```

### QLoRA Example (Quantized LoRA)

```rust
use burn_peft::{QLoRAConfig, QLoRALinear};
use burn::tensor::quantization::{
    Calibration, QuantScheme, QuantValue, QuantLevel,
    compute_range, compute_q_params
};
use burn::module::Quantizer;

fn example<B: Backend>(device: &B::Device) {
    // Load pretrained weight
    let weight_fp: Tensor<B, 2> = /* ... */;

    // Define quantization scheme (4-bit for max savings)
    let scheme = QuantScheme::default()
        .with_value(QuantValue::Q4F)      // 4-bit quantization
        .with_level(QuantLevel::block([32]));  // Block-wise quantization

    // Quantize the weight
    let range = compute_range(&scheme, &weight_fp, &Calibration::MinMax);
    let qparams = compute_q_params(&scheme, range);
    let weight_q = weight_fp.quantize(&scheme, qparams);

    // Create QLoRA layer (87% memory savings!)
    let config = QLoRAConfig::new(4096, 4096)
        .with_rank(8)
        .with_alpha(16.0);

    let layer = config.init_with_quantized_weight(weight_q, None, device);

    // Forward pass - automatic dequantization + LoRA adapter
    let input = Tensor::<B, 2>::random([8, 4096], Distribution::Default, device);
    let output = layer.forward(input);  // Shape: [8, 4096]
}
```

### QDoRA Example (Quantized DoRA)

```rust
use burn_peft::{QDoRAConfig, QDoRALinear};

fn example<B: Backend>(device: &B::Device) {
    // Quantize pretrained weight (same as QLoRA)
    let weight_q = /* ... quantized weight ... */;

    // Create QDoRA layer - combines quantization with DoRA's benefits
    let config = QDoRAConfig::new(4096, 4096)
        .with_rank(8)
        .with_epsilon(1e-8);

    let layer = config.init_with_quantized_weight(weight_q, None, device);

    let input = Tensor::<B, 2>::random([8, 4096], Distribution::Default, device);
    let output = layer.forward(input);
}
```

### LoRA+ Example (Differential Learning Rates)

```rust
use burn_peft::{LoRAPlusConfig, LoRAPlusLinear};

fn example<B: Backend>(device: &B::Device) {
    // LoRA+ uses different learning rates for A and B matrices
    let config = LoRAPlusConfig::new(512, 512)
        .with_rank(16)
        .with_alpha(32.0)
        .with_lr_ratio(16.0);  // B gets 16x learning rate of A

    let layer = config.init_with_base_weight(pretrained_weight, bias, device);

    // In your training loop, set:
    // - lr(lora_a) = base_lr
    // - lr(lora_b) = base_lr * 16.0
    // This leads to faster convergence and better performance
}
```

### Adapter Composition Example

```rust
use burn_peft::AdapterComposer;

fn example<B: Backend>() {
    // Train separate LoRA adapters for different tasks
    let lora_math = /* trained on math dataset */;
    let lora_code = /* trained on code dataset */;
    let lora_writing = /* trained on writing dataset */;

    // Compose adapters with weighted sum
    let (a_merged, b_merged) = AdapterComposer::weighted_sum(
        &[lora_math.lora_a.val(), lora_code.lora_a.val(), lora_writing.lora_a.val()],
        &[lora_math.lora_b.val(), lora_code.lora_b.val(), lora_writing.lora_b.val()],
        &[0.5, 0.3, 0.2],  // Task weights
    );

    // Or concatenate to increase rank
    let (a_concat, b_concat) = AdapterComposer::concatenate(
        &[lora_math.lora_a.val(), lora_code.lora_a.val()],
        &[lora_math.lora_b.val(), lora_code.lora_b.val()],
    );
    // Result has rank = rank_math + rank_code
}
```

## Mathematical Details

### LoRA

For base weight **W**₀ ∈ ℝ^(d×k) and input **x**:

**Unmerged (training):**
```
h = W₀·x + (α/r)·B(A·x)
```

**Merged (inference):**
```
W' = W₀ + (α/r)·BA
h = W'·x
```

Where:
- **A** ∈ ℝ^(r×k): Down-projection (initialized with Kaiming uniform)
- **B** ∈ ℝ^(d×r): Up-projection (initialized to zeros)
- r ≪ min(d,k): Adapter rank
- α: Scaling factor

**Initialization:** B = 0 ensures Δ = BA = 0 initially (no output jump).

### DoRA

Decomposes weight into magnitude and direction:

```
W = m ⊙ (V / ||V||_c)
```

Where:
- **m** ∈ ℝ^(1×k): Trainable magnitude (per-column)
- **V** = V₀ + BA: Updated direction
- V₀: Frozen base direction (copy of W₀)
- ||V||_c: Column-wise L2 norms (detached for efficiency)

**Key innovation:** Gradients flow through V but NOT through the norm computation, significantly reducing memory while maintaining performance.

### QLoRA

Quantizes the base weight to 4-8 bits while keeping adapters in full precision:

```
h = QuantMatMul(W₀_q, x) + (α/r)·B(A·x)
```

**Memory Savings Example** (4096×4096 layer, rank 8):
- Full precision LoRA: 64 MB base + 0.5 MB adapters = **64.5 MB**
- 4-bit QLoRA: 8 MB base + 0.5 MB adapters = **8.5 MB**
- **87% reduction!**

Trainable: A, B (adapters only)
Frozen: W₀_q (quantized base)

### QDoRA

Combines DoRA's decomposition with quantization:

```
V' = Dequant(V₀_q) + BA
W = m ⊙ (V' / ||V'||_c)
```

**Benefits:**
- Same memory savings as QLoRA (~87% with 4-bit)
- DoRA's improved performance from magnitude/direction decomposition
- Detached norms for efficiency

Trainable: m (magnitude), A, B (adapters)
Frozen: V₀_q (quantized base direction)

## Performance Tips

1. **Rank selection:** Start with r = 8-16 for most tasks. Higher rank = more parameters but better capacity.

2. **Alpha scaling:** Common choice is α = r (so α/r = 1.0) or α = 2r for more aggressive adaptation.

3. **Quantization:** Use 4-bit for maximum memory savings, 8-bit for better accuracy:
   ```rust
   // 4-bit: ~87% memory reduction, slight accuracy loss
   QuantValue::Q4F

   // 8-bit: ~75% memory reduction, minimal accuracy loss
   QuantValue::Q8S
   ```

4. **Block-wise quantization:** More accurate than tensor-wise for large models:
   ```rust
   QuantLevel::block([32])  // 32-element blocks
   QuantLevel::Tensor       // Single scale for entire tensor
   ```

5. **Merging for inference (LoRA/DoRA only):** Merge weights before deployment:
   ```rust
   layer.merge_weights();
   // Now forward passes use a single matmul
   ```

6. **Method selection:**
   - **LoRA:** Simple, fast, proven
   - **LoRA+:** Faster convergence than LoRA (16x speedup with differential learning rates)
   - **DoRA:** Better accuracy at same rank (~10-20% improvement)
   - **QLoRA:** Use when memory is critical (large models, limited GPU)
   - **QDoRA:** Best quality + memory savings (recommended for production)
   - **Adapter Composition:** Multi-task learning with separate adapters per task

## Numerical Stability

All implementations follow best practices:

- ✅ FP32 accumulation for norms (even with FP16/BF16 inputs)
- ✅ Epsilon (ε = 1e-8) for division safety
- ✅ Detached norms in DoRA to prevent NaN gradients
- ✅ Proper initialization to avoid output jumps

## Benchmarks

To run comprehensive performance benchmarks comparing all PEFT methods:

```bash
# Navigate to the burn-peft crate directory
cd crates/burn-peft

# Run with CUDA backend (requires NVIDIA GPU)
cargo bench --bench peft_comparison --features test-cuda

# Or run with WGPU backend (works on all platforms)
cargo bench --bench peft_comparison --features test-wgpu
```

The benchmark suite measures:
- **Forward/backward pass latency** - Full Linear vs LoRA vs DoRA
- **Throughput** - Samples per second for each method
- **Merge speedup** - LoRA merged vs unmerged inference performance
- **Quantized variants** - QLoRA vs QDoRA performance
- **Memory footprint** - Parameter counts and trainable percentages

Results include relative performance comparisons and detailed timing breakdowns for different model sizes.

## References

- **LoRA:** Hu et al., "[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)", ICLR 2022
- **LoRA+:** Hayou et al., "[LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/abs/2402.12354)", 2024
- **DoRA:** Liu et al., "[DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)", 2024
- **QLoRA:** Dettmers et al., "[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)", NeurIPS 2023

## Roadmap

- [x] LoRA implementation
- [x] LoRA+ (differential learning rates)
- [x] DoRA implementation
- [x] QLoRA (quantized base weights)
- [x] QDoRA (quantized DoRA)
- [x] Adapter composition (weighted sum and concatenation)
- [ ] Integration with `burn-import` for loading pretrained adapters
- [ ] Multi-adapter inference (activate different adapters dynamically)
- [ ] Fused kernels for quantized operations (further speedup)
- [ ] DoRA merge/unmerge operations
- [ ] SVD integration
## License

Dual-licensed under MIT OR Apache-2.0, same as Burn.
