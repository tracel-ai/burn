# LoRA Fine-tuning Example

This example demonstrates how to use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning in Burn.

## What is LoRA?

LoRA is a technique for efficiently fine-tuning large pre-trained models by:

1. **Freezing** the original model weights
2. **Adding** small trainable low-rank matrices (A and B) to specific layers
3. **Computing**: `output = base(x) + (x @ A @ B) * scaling`

This dramatically reduces the number of trainable parameters while maintaining model quality.

## Key Benefits

- **Parameter Efficiency**: Train only ~1-5% of original parameters
- **Memory Efficient**: Smaller optimizer states and gradients
- **Mergeable**: LoRA weights can be merged into base model for zero inference overhead
- **Composable**: Multiple LoRA adaptations can be swapped without reloading base model

## Running the Example

```bash
# Using ndarray backend (CPU)
cargo run --example lora --release --features ndarray

# Using wgpu backend (GPU)
cargo run --example lora --release --features wgpu

# Using LibTorch CPU
cargo run --example lora --release --features tch-cpu

# Using LibTorch GPU
cargo run --example lora --release --features tch-gpu
```

## Example Output

```
=== LoRA Fine-tuning Example ===

Base model parameters: 99850

LoRA Configuration:
  Rank: 8
  Alpha: 16
  Scaling: 2.0000
  Bias mode: None

LoRA trainable parameters: 6144
Parameter reduction: 93.85%

LoRA model output shape: [4, 10]
Merged model output shape: [4, 10]

Max difference between LoRA and merged outputs: 0.00e+00
SUCCESS: LoRA and merged outputs match!

Sample output (first row):
...

=== Example Complete ===
```

## Code Highlights

### Applying LoRA to a Model

```rust
use burn::nn::lora::{LoraConfig, LoraAdaptable, LoraBias};

// Configure LoRA
let config = LoraConfig::new(8)      // rank = 8
    .with_alpha(16.0)                 // scaling = 16/8 = 2.0
    .with_dropout(0.1)                // optional dropout
    .with_bias(LoraBias::All);        // train biases in LoRA layers

// Apply to a linear layer
let lora_linear = linear_layer.with_lora(&config, &device);
```

### Merging for Inference

```rust
// After training, merge LoRA weights into base layer
let merged_linear = lora_linear.merge();

// merged_linear is a regular Linear with no LoRA overhead
// Output is identical to lora_linear (ignoring dropout)
```

## Configuration Options

| Option       | Description                       | Default   |
| ------------ | --------------------------------- | --------- |
| `rank`       | Low-rank dimension (4-64 typical) | Required  |
| `alpha`      | Scaling numerator                 | 1.0       |
| `dropout`    | Dropout on LoRA branch            | 0.0       |
| `bias`       | Bias training mode                | `None`    |
| `use_rslora` | Rank-stabilized scaling           | false     |
| `init`       | Weight initialization             | `Kaiming` |

## Bias Modes

- `LoraBias::None` - All biases frozen (default)
- `LoraBias::All` - Biases trainable in LoRA-wrapped layers
