# LoRA (Low-Rank Adaptation)

LoRA enables parameter-efficient fine-tuning by freezing pre-trained weights and adding small
trainable low-rank matrices. This technique is particularly useful when:

- You have a large pre-trained model and limited GPU memory
- You want to create multiple specialized adapters from one base model
- You need fast training with reduced parameter count

All the code can be found under the
[examples directory](https://github.com/tracel-ai/burn/tree/main/examples/lora-finetuning).

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Rank** | Size of low-rank matrices (smaller = fewer parameters, typical: 4-64) |
| **Alpha** | Scaling factor for LoRA contribution (typical: 2x rank) |
| **Scaling** | Computed as `alpha / rank`, controls how much LoRA affects output |
| **Adapter** | The trainable LoRA weights that can be saved/loaded independently |

## Basic Usage

### Configuration

Configure LoRA with rank, alpha, and optional dropout:

```rust,ignore
use burn::nn::lora::{LoraConfig, LoraBias};

// Configure LoRA
let lora_config = LoraConfig::new(4)   // rank = 4
    .with_alpha(8.0)                    // scaling = alpha/rank = 2.0
    .with_dropout(0.0)
    .with_bias(LoraBias::None);
```

### Applying LoRA to Layers

Apply LoRA to any `Linear` layer using the `LoraAdaptable` trait:

```rust,ignore
use burn::nn::lora::LoraAdaptable;

// Apply LoRA to a linear layer
let lora_linear = linear.with_lora(&lora_config, &device);

// The base weights are frozen, only LoRA matrices are trainable
let output = lora_linear.forward(input);
```

### Merging for Inference

After training, merge LoRA weights back into the base layer for zero-overhead inference:

```rust,ignore
// Merge LoRA into base weights (zero inference overhead)
let merged_linear = lora_linear.merge();

// Now it's a regular Linear layer with updated weights
let output = merged_linear.forward(input);
```

## Configuration Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `rank` | `usize` | Low-rank dimension | Required |
| `alpha` | `f64` | Scaling factor | `1.0` |
| `dropout` | `f64` | Dropout probability on LoRA branch | `0.0` |
| `bias` | `LoraBias` | Bias training mode | `None` |

### Bias Modes

The `LoraBias` enum controls how biases are handled in LoRA-wrapped layers:

- `LoraBias::None` - Keep bias frozen (default, most common)
- `LoraBias::All` - Unfreeze bias in LoRA-wrapped layers (for non-LoRA layers, manually unfreeze)

## Adapter Persistence

One of LoRA's key benefits is the ability to save and load adapters independently of the base
model. This enables:

- Sharing small adapter files instead of full model weights
- Swapping adapters at runtime for different tasks
- Efficient storage of multiple fine-tuned variants

### Saving Adapters

```rust,ignore
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

// Save individual adapter (~KB instead of MB/GB for full model)
lora_linear.save_adapter("./my-adapter/layer1", &recorder)?;
```

### Loading Adapters

```rust,ignore
// Create fresh model with LoRA applied
let fresh_lora = linear.with_lora(&lora_config, &device);

// Load trained adapter weights
let loaded = fresh_lora.load_adapter_file(
    "./my-adapter/layer1",
    &recorder,
    &device
)?;
```

## Full Example: LoRA Fine-tuning

Here's the complete workflow demonstrated in the example:

### 1. Define a Model with LoRA

```rust,ignore
use burn::nn::lora::{LoraConfig, LoraLinear, LoraAdaptable};
use burn::nn::{Linear, LinearConfig, Relu};
use burn::prelude::*;

// Base model
#[derive(Module, Debug)]
pub struct SimpleMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    relu: Relu,
}

// Model with LoRA applied to fc1 and fc2
#[derive(Module, Debug)]
pub struct SimpleMlpWithLora<B: Backend> {
    fc1: LoraLinear<B>,
    fc2: LoraLinear<B>,
    fc3: Linear<B>,  // Keep output layer frozen without LoRA
    relu: Relu,
}

// Apply LoRA to a base model
pub fn apply_lora<B: Backend>(
    model: SimpleMlp<B>,
    config: &LoraConfig,
    device: &B::Device,
) -> SimpleMlpWithLora<B> {
    SimpleMlpWithLora {
        fc1: model.fc1.with_lora(config, device),
        fc2: model.fc2.with_lora(config, device),
        fc3: model.fc3.no_grad(), // Freeze without LoRA
        relu: model.relu,
    }
}
```

### 2. Train with SupervisedTraining

```rust,ignore
use burn::train::{SupervisedTraining, Learner};
use burn::optim::AdamConfig;

// Apply LoRA to pre-trained model
let lora_model = apply_lora(base_model, &lora_config, &device);

// Setup training
let training = SupervisedTraining::new(artifact_dir, train_loader, valid_loader)
    .metric_train_numeric(LossMetric::new())
    .metric_valid_numeric(LossMetric::new())
    .num_epochs(100)
    .summary();

// Train (only LoRA parameters are updated)
let result = training.run(Learner::new(
    lora_model,
    AdamConfig::new().init(),
    1e-3,
));
```

### 3. Save and Load Adapters

```rust,ignore
// Save adapters after training
lora_model.save_adapters("./adapters")?;

// Later: Load base model and apply trained adapters
let fresh_base = SimpleMlpConfig::new(32, 64, 1).init::<B>(&device);
let fresh_lora = apply_lora(fresh_base, &lora_config, &device);
let loaded_model = fresh_lora.load_adapters("./adapters", &device)?;
```

### 4. Merge for Deployment

```rust,ignore
// Merge LoRA weights into base model
let merged_model: SimpleMlp<B> = lora_model.merge();

// Deploy with no LoRA overhead
let output = merged_model.forward(input);
```

## Running the Example

```bash
# Run with ndarray backend (CPU)
cargo run -p lora-finetuning --features ndarray --example lora

# Run with wgpu backend (GPU)
cargo run -p lora-finetuning --features wgpu --example lora
```

The example demonstrates:
- Creating a base MLP model
- Applying LoRA to hidden layers
- Training with a TUI dashboard
- Saving/loading adapters from disk
- Merging weights for inference

## When to Use LoRA

| Use Case | LoRA Benefits |
|----------|--------------|
| Large language model fine-tuning | Reduces memory from GBs to MBs |
| Multiple task-specific models | Share base weights, swap small adapters |
| Edge deployment | Train on cloud, deploy merged weights |
| Experimentation | Quick iteration with small parameter sets |

## Comparison: Full Fine-tuning vs LoRA

| Aspect | Full Fine-tuning | LoRA |
|--------|-----------------|------|
| Trainable params | All | ~0.1-1% |
| Memory usage | High | Low |
| Training speed | Slower | Faster |
| Checkpoint size | Full model | Adapter only |
| Inference overhead | None | None (after merge) |
