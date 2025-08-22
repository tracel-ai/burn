# Model Checks

This directory contains model verification and validation tests for burn-import. Each subdirectory
represents a different model that we test to ensure burn-import can correctly:

1. Import ONNX models
2. Generate Rust code from the models
3. Build and run the generated code

## Purpose

The model-checks serve as integration tests to verify that burn-import works correctly with
real-world models. These tests help catch regressions and ensure compatibility with various ONNX
operators and model architectures.

## Structure

Each model directory typically contains:

- Model download/preparation script (e.g., `get_model.py`)
- `build.rs` - Build script that uses burn-import to generate Rust code
- `src/main.rs` - Test code that runs the generated model
- `Cargo.toml` - Package configuration
- `artifacts/` - Directory for downloaded ONNX models (created by the script)

Generated files (not tracked in git):

- `target/` - Build artifacts and generated model code

## Two-Step Process

### Step 1: Download and Prepare the Model

First, download the model and convert it to the required ONNX format:

```bash
cd model-checks/<model-name>
python get_model.py
# or using uv:
uv run get_model.py
```

The model preparation script typically:

- Downloads the model (if not already present)
- Converts it to ONNX format with the appropriate opset version
- Validates the model structure
- Saves the prepared model to `artifacts/`

Scripts are designed to skip downloading if the ONNX model already exists, saving time and
bandwidth.

### Step 2: Build and Run the Model

Once the ONNX model is ready, build and run the Rust code:

```bash
cargo build
cargo run
```

The build process will:

- Check that the ONNX model exists (with helpful error messages if not)
- Generate Rust code from the ONNX model using burn-import
- Compile the generated code
