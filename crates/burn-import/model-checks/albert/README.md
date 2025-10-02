# ALBERT Model Checks

This crate provides a unified interface for testing ALBERT model variants with Burn.

## Supported Models

- `albert-base-v2` - ALBERT Base v2 from HuggingFace (https://huggingface.co/albert/albert-base-v2)

## Usage

### 1. Download and prepare a model

```bash
# Using Python directly
python get_model.py --model albert-base-v2

# Or using uv
uv run get_model.py --model albert-base-v2

# List available models
uv run get_model.py --list
```

### 2. Build and run the model test

```bash
# Build the model
ALBERT_MODEL=albert-base-v2 cargo build

# Run the test
ALBERT_MODEL=albert-base-v2 cargo run --release
```

## Directory Structure

```
albert/
├── artifacts/           # Downloaded ONNX models and test data
│   ├── albert-base-v2_opset16.onnx
│   ├── albert-base-v2_test_data.pt
│   └── ...
├── src/
│   └── main.rs         # Test runner
├── build.rs            # Build script that generates model code
├── get_model.py        # Model download and preparation script
└── Cargo.toml
```

## Model Architecture

ALBERT (A Lite BERT) is a lighter version of BERT that uses parameter-sharing techniques to reduce
model size while maintaining performance. The model has:

- **Inputs**:
  - `input_ids`: Token IDs (shape: [batch_size, sequence_length])
  - `attention_mask`: Attention mask (shape: [batch_size, sequence_length])
  - `token_type_ids`: Token type IDs (shape: [batch_size, sequence_length])

- **Outputs**:
  - `last_hidden_state`: Sequence of hidden states (shape: [batch_size, sequence_length,
    hidden_size])
  - `pooler_output`: Pooled output for classification tasks (shape: [batch_size, hidden_size])

## Notes

- The default sequence length is 128 tokens
- ALBERT Base v2 has a hidden size of 768
- The model uses ONNX opset 16
- Test data is generated with random inputs for reproducibility (seed=42)
