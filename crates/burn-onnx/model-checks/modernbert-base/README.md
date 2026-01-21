# ModernBERT-base Model Check

This crate provides testing for the ModernBERT-base model with Burn.

## Model

- `ModernBERT-base` - Modern BERT variant from Answer.AI
  (https://huggingface.co/answerdotai/ModernBERT-base)

## Usage

### 1. Download and prepare the model

```bash
# Using Python directly
python get_model.py

# Or using uv
uv run get_model.py
```

**Note:** This will download the model from HuggingFace and export it to ONNX format using PyTorch.
Make sure you have the `transformers` library installed.

### 2. Build and run the model test

```bash
# Build the model
cargo build

# Run the test
cargo run --release
```

## Directory Structure

```
modernbert-base/
├── artifacts/                         # Downloaded ONNX model and test data
│   ├── modernbert-base_opset16.onnx
│   ├── test_data.pt
│   └── model-python.txt
├── src/
│   └── main.rs                        # Test runner
├── build.rs                           # Build script that generates model code
├── get_model.py                       # Model download and ONNX export script
├── Cargo.toml
└── README.md
```

## Model Architecture

ModernBERT is a modern variant of BERT designed by Answer.AI with improved efficiency and
performance. It features several architectural improvements over the original BERT, including better
positional embeddings and optimized attention mechanisms.

- **Inputs**:
  - `input_ids`: Token IDs (shape: [batch_size, sequence_length])
  - `attention_mask`: Attention mask (shape: [batch_size, sequence_length])

- **Outputs**:
  - `last_hidden_state`: Sequence of hidden states (shape: [batch_size, sequence_length, 768])
  - `pooled_output`: Mean-pooled sentence embeddings (computed, shape: [batch_size, 768])

## Notes

- The default sequence length is 512 tokens
- The model has a hidden size of 768
- The model uses ONNX opset 16
- Test data is generated with random inputs for reproducibility (seed=42)
- Vocabulary size: 50,368 tokens
- The pooled output is computed using mean pooling over the last hidden state
