# Safetensors Model

## Introduction

Burn supports importing model weights from the Safetensors format, a secure and efficient
alternative to pickle-based formats. Whether you've trained your model in PyTorch or you want to use
a pre-trained model that provides weights in Safetensors format, you can easily import them into
Burn.

This guide demonstrates the complete workflow for exporting models to Safetensors format and
importing them into Burn.

## Exporting Models to Safetensors Format

To export a PyTorch model to Safetensors format, you'll need the `safetensors` Python library. This
library provides a simple API for saving model weights in the Safetensors format.

### Example: Exporting a PyTorch Model

```python
import torch
import torch.nn as nn
from safetensors.torch import save_file

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, (2,2))
        self.conv2 = nn.Conv2d(2, 2, (2,2), bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Initialize model and ensure it's on CPU
    model = Net().to(torch.device("cpu"))

    # Extract model weights dictionary
    model_weights = model.state_dict()

    # Save to Safetensors format
    save_file(model_weights, "conv2d.safetensors")
```

### Verifying the Export

You can verify your exported model by viewing the `.safetensors` file in
[Netron](https://github.com/lutzroeder/netron), a neural network visualization tool. A correctly
exported file will display a flat structure of tensors, similar to a PyTorch `.pt` weights file.

## Importing Safetensors Models into Burn

Importing a Safetensors model into Burn involves two main steps:

1. Defining the model architecture in Burn
2. Loading the weights from the Safetensors file

### Step 1: Define the Model in Burn

First, you need to create a Burn model that matches the architecture of the model you exported:

```rust
use burn::{
    nn::conv::{Conv2d, Conv2dConfig},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
}

impl<B: Backend> Net<B> {
    /// Create a new model.
    pub fn init(device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([2, 2], [2, 2])
            .init(device);
        let conv2 = Conv2dConfig::new([2, 2], [2, 2])
            .with_bias(false)
            .init(device);
        Self { conv1, conv2 }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(x);
        self.conv2.forward(x)
    }
}
```

### Step 2: Load the Weights

You have two options for loading the weights:

#### Option A: Load Dynamically at Runtime

This approach loads the Safetensors file directly at runtime, requiring the `burn-import`
dependency:

```rust
use crate::model;
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::safetensors::SafetensorsFileRecorder;

type Backend = burn_ndarray::NdArray<f32>;

fn main() {
    let device = Default::default();

    // Load weights from Safetensors file
    let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
        .load("./conv2d.safetensors".into(), &device)
        .expect("Should decode state successfully");

    // Initialize model and load weights
    let model = model::Net::<Backend>::init(&device).load_record(record);
}
```

#### Option B: Pre-convert to Burn's Binary Format

This approach converts the Safetensors file to Burn's optimized binary format during build time,
removing the runtime dependency on `burn-import`:

```rust
// This code would go in build.rs or a separate tool

use crate::model;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_import::safetensors::SafetensorsFileRecorder;

type Backend = burn_ndarray::NdArray<f32>;

fn convert_model() {
    let device = Default::default();

    // Load from Safetensors
    let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::default();
    let record = recorder
        .load("./conv2d.safetensors".into(), &device)
        .expect("Should decode state successfully");

    // Save to Burn's binary format
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    recorder
        .record(record, "model.mpk".into())
        .expect("Failed to save model record");
}

// In your application code
fn load_model() -> Net<Backend> {
    let device = Default::default();

    // Load from Burn's binary format
    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
        .load("./model.mpk".into(), &device)
        .expect("Should decode state successfully");

    Net::<Backend>::init(&device).load_record(record)
}
```

> **Note**: For examples of pre-converting models, see the `examples/import-model-weights` directory
> in the Burn repository.

## Advanced Configuration Options

### Framework-Specific Adapters

When importing Safetensors models, you can specify an adapter type to handle framework-specific
tensor transformations. This is crucial when importing models from different ML frameworks, as
tensor layouts and naming conventions can vary:

```rust
use burn_import::safetensors::{AdapterType, LoadArgs, SafetensorsFileRecorder};
use burn::record::{FullPrecisionSettings, Recorder};

let device = Default::default();

// Create load arguments with framework-specific adapter
let load_args = LoadArgs::new("model.safetensors".into())
    .with_adapter_type(AdapterType::PyTorch); // Default adapter

// Load with the specified adapter
let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
    .load(load_args, &device)
    .expect("Should decode state successfully");
```

#### Available Adapter Types

| Adapter Type          | Description                                                                                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PyTorch** (default) | Automatically applies PyTorch-specific transformations:<br>- Transposes weights for linear layers<br>- Renames normalization parameters (weight→gamma, bias→beta) |
| **NoAdapter**         | Loads tensors directly without any transformations<br>- Useful when importing from frameworks that already match Burn's tensor layout                             |
| **TensorFlow**        | Reserved for future implementation                                                                                                                                |

## Troubleshooting and Advanced Features

### Key Remapping for Different Model Architectures

If your Burn model structure doesn't match the parameter names in the Safetensors file, you can
remap keys using regular expressions:

```rust
use burn_import::safetensors::{LoadArgs, SafetensorsFileRecorder};
use burn::record::{FullPrecisionSettings, Recorder};

let device = Default::default();

// Create load arguments with key remapping
let load_args = LoadArgs::new("model.safetensors".into())
    // Remove "conv" prefix, e.g. "conv.conv1" -> "conv1"
    .with_key_remap("conv\\.(.*)", "$1");

let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
    .load(load_args, &device)
    .expect("Should decode state successfully");

let model = Net::<Backend>::init(&device).load_record(record);
```

### Debugging with Key Inspection

To help with troubleshooting import issues, you can enable debugging to print the original and
remapped keys:

```rust
use burn_import::safetensors::{LoadArgs, SafetensorsFileRecorder};
use burn::record::{FullPrecisionSettings, Recorder};

let device = Default::default();

// Enable debug printing of keys
let load_args = LoadArgs::new("model.safetensors".into())
    .with_key_remap("conv\\.(.*)", "$1")
    .with_debug_print();  // Print original and remapped keys

let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
    .load(load_args, &device)
    .expect("Should decode state successfully");
```

### Automatic Handling of Non-Contiguous Indices

The SafetensorsFileRecorder automatically handles non-contiguous indices in model layer names. For
example, if the source model contains indices with gaps:

```
"model.layers.0.weight"
"model.layers.0.bias"
"model.layers.2.weight"  // Note the gap (no index 1)
"model.layers.2.bias"
"model.layers.4.weight"
"model.layers.4.bias"
```

The recorder will automatically reindex these to be contiguous while preserving their order:

```
"model.layers.0.weight"
"model.layers.0.bias"
"model.layers.1.weight"  // Reindexed from 2
"model.layers.1.bias"
"model.layers.2.weight"  // Reindexed from 4
"model.layers.2.bias"
```

### Partial Model Loading

You can selectively load weights into a partial model, which is useful for:

- Loading only the encoder from an encoder-decoder architecture
- Fine-tuning specific layers while initializing others randomly
- Creating hybrid models combining parts from different sources

### Support for Enum Modules

The SafetensorsFileRecorder supports models containing enum modules with new-type variants. The enum
variant is automatically selected based on the enum variant type, allowing for flexible model
architectures.
