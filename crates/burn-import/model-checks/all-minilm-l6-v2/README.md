# all-MiniLM-L6-v2 Model Check

This crate provides testing for the all-MiniLM-L6-v2 sentence transformer model with Burn.

## Model

- `all-MiniLM-L6-v2` - Sentence transformer model from HuggingFace
  (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## Usage

### 1. Download and prepare the model

```bash
# Using Python directly
python get_model.py

# Or using uv
uv run get_model.py
```

### 2. Build and run the model test

```bash
# Build the model
cargo build

# Run the test
cargo run --release
```

## Directory Structure

```
all-minilm-l6-v2/
├── artifacts/                         # Downloaded ONNX model and test data
│   ├── all-minilm-l6-v2_opset16.onnx
│   ├── test_data.pt
│   └── model-python.txt
├── src/
│   └── main.rs                        # Test runner
├── build.rs                           # Build script that generates model code
├── get_model.py                       # Model download and preparation script
├── Cargo.toml
└── README.md
```

## Model Architecture

all-MiniLM-L6-v2 is a sentence transformer model based on Microsoft's MiniLM architecture. It maps
sentences and paragraphs to a 384-dimensional dense vector space and is commonly used for semantic
search, clustering, and similarity tasks.

- **Inputs**:
  - `input_ids`: Token IDs (shape: [batch_size, sequence_length])
  - `attention_mask`: Attention mask (shape: [batch_size, sequence_length])
  - `token_type_ids`: Token type IDs (shape: [batch_size, sequence_length])

- **Outputs**:
  - `last_hidden_state`: Sequence of hidden states (shape: [batch_size, sequence_length, 384])

## Notes

- The default sequence length is 128 tokens
- The model has a hidden size of 384
- The model uses ONNX opset 16
- Test data is generated with random inputs for reproducibility (seed=42)
- For sentence embeddings, you typically use mean pooling on the last_hidden_state
