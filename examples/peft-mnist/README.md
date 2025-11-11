# PEFT MNIST Example

Demonstrates parameter-efficient fine-tuning (PEFT) using LoRA on MNIST with Burn.

## What It Does

1. **Creates a baseline MLP** on MNIST (784 → 512 → 256 → 10)
2. **Converts to LoRA** with frozen base weights and trainable adapters
3. **Trains for 3 epochs** with only ~10% of parameters trainable
4. **Saves the model** for inference

## Running the Example

### Training

```bash
# Default backend (ndarray/CPU)
cargo run --release

# CUDA backend
cargo run --release --features cuda

# WGPU backend
cargo run --release --features wgpu

# Explicitly train
cargo run --release -- train
```

### Inference

```bash
# Run inference on test samples
cargo run --release -- infer

# With CUDA
cargo run --release --features cuda -- infer
```

### Benchmarks

```bash
# Run LoRA vs full fine-tuning benchmarks
cargo run --release -- bench

# With CUDA for faster execution
cargo run --release --features cuda -- bench
```

## What You'll See

**During Training:**
```
🔥 Burn PEFT MNIST Example
========================================
📦 Loading MNIST dataset...
📊 Creating baseline MLP...
   Baseline parameters: 500234

🎯 Converting to LoRA (rank=8, alpha=16)...
   LoRA total parameters: 527642
   Parameter reduction: -5.5%

🚀 Training LoRA model for 3 epochs...
[Training metrics and progress...]

✅ Training complete!
   Model saved to: ./tmp/peft-mnist
```

**During Inference:**
```
🔥 Burn PEFT Inference
========================================
Loading trained LoRA model (rank=8, alpha=16)...

Running inference on 10 test samples...

Sample 1: Predicted: 7, Actual: 7, ✓
Sample 2: Predicted: 2, Actual: 2, ✓
...

Accuracy on 10 samples: 9/10 (90.0%)
```

**During Benchmarks:**
```
🔥 Burn PEFT Benchmark: LoRA vs Full Fine-Tuning
================================================

📊 Small Model: 784 → 512 → 256 → 10

  Parameters:
    Full Model:            500234
    LoRA Total:            527642
    LoRA Trainable:         15912  (3.2% of full)
    LoRA Frozen:           511730  (102.3% of full)

  Memory (FP32):
    Full Model:            1.91 MB
    LoRA Total:            2.01 MB
    LoRA Trainable:        0.06 MB  (only these need gradients)
    Gradient Memory:
      Full Training:       1.91 MB  (all params)
      LoRA Training:       0.06 MB  (adapters only)
      Savings:             1.85 MB  (96.8%)

  Forward Pass (1000 iterations):
    Full:              245ms
    LoRA (unmerged):   267ms  (1.09x)
    LoRA (merged):     241ms  (0.98x)

  Backward Pass (100 iterations):
    Full:              1.23s
    LoRA:              1.18s  (0.96x)

📊 Medium Model: 784 → 2048 → 1024 → 10

  Parameters:
    Full Model:          3737610
    LoRA Trainable:        51288  (1.4% of full)

  Memory (FP32):
    Full Model:           14.26 MB
    LoRA Trainable:        0.20 MB
    Gradient Savings:     14.06 MB  (98.6%)

  QLoRA (4-bit) Memory:
    Base (4-bit):          1.78 MB  (quantized)
    Adapters (FP32):       0.20 MB
    Total:                 1.98 MB
    vs Full Model:        12.28 MB saved (86.1%)
```

## Methods Demonstrated

This example shows:
- **LoRA (Low-Rank Adaptation)**: Freeze base weights, add low-rank adapters
- **Converting pretrained models** to PEFT format
- **Training with Burn's learner** framework
- **Saving and loading** trained PEFT models
- **Inference** with loaded models

## Key Takeaways

1. **Parameter Efficiency**: Only adapter matrices (lora_a, lora_b) are trainable
2. **Drop-in Replacement**: LoRALinear can replace any Linear layer
3. **Easy Inference**: Load config + weights, run forward pass
4. **Flexibility**: Configure rank and alpha for different compression ratios

## Configuration

Edit `src/training.rs` to customize:
```rust
#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub num_epochs: usize,      // default: 3
    pub batch_size: usize,      // default: 64
    pub learning_rate: f64,     // default: 1e-3
    pub lora_rank: usize,       // default: 8
    pub lora_alpha: f64,        // default: 16.0
}
```

## Advanced Features

For more PEFT methods, see the `burn-peft` crate:
- **DoRA**: Weight-Decomposed LoRA
- **QLoRA**: Quantized LoRA (4-bit base weights, 87% memory savings)
- **QDoRA**: Quantized DoRA
- **LoRA+**: Differential learning rates
- **Adapter Composition**: Merge multiple LoRA adapters
