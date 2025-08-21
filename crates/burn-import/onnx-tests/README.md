# ONNX Tests

This crate contains ONNX models used for testing the conversion process from ONNX to Burn source
code through the `burn-import` crate. These tests are designed as end-to-end tests, ensuring that
ONNX models are accurately converted into Burn source code that compiles without errors and produces
the same outputs as the original ONNX model.

## Directory Structure

- `tests/<op_name>/`: Each operator or model has its own directory
- `tests/<op_name>/<op_name>.py`: Python script that generates the ONNX model
- `tests/<op_name>/<op_name>.onnx`: Generated ONNX model
- `tests/<op_name>/mod.rs`: Test implementation for the specific operator
- `tests/test_mod.rs`: Main test file that integrates all operator tests
- `build.rs`: Build script that generates ONNX models before running tests

## Setting Up Your Python Environment

### Using uv (Recommended)

You can use [`uv`](https://docs.astral.sh/uv/) to set up a Python environment with the necessary
dependencies:

```sh
cd crates/burn-import/onnx-tests
uv sync # or uv sync -f
```

This will create a `.venv` directory with all the required dependencies.

### Manual Setup

If you prefer to set up manually, you need to install the following packages:

```sh
pip install onnx==1.15.0 torch==2.1.1
```

Additional dependencies are specified in `requirements.lock`.

## Creating a Test for a New Operator

There are two main approaches to generating ONNX files for testing:

1. **Exporting a model from PyTorch** (most common)
2. **Constructing an ONNX graph directly** (for specific cases)

### 1. Create the Python Script

Create a new directory and Python script:

```sh
mkdir -p tests/my_new_op
touch tests/my_new_op/my_new_op.py
```

#### Approach 1: Exporting a PyTorch Model to ONNX

Your script should:

- Import the necessary PyTorch modules
- Define a model that uses your operator
- Generate test inputs
- Export the model to ONNX format
- Run the model on test inputs and print the output

Example structure:

```python
import torch
import torch.nn as nn
import torch.onnx

# Define a simple model that uses your operator
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ...

    def forward(self, x):
        # Use your operator here
        return my_operation(x)

# Create an instance of the model
model = MyModel()

# Generate test input
input_tensor = torch.randn(1, 3, 224, 224)

# Export the model to ONNX
torch.onnx.export(
    model,
    input_tensor,
    "tests/my_new_op/my_new_op.onnx",
    opset_version=16,
    input_names=["input"],
    output_names=["output"],
    do_constant_folding=False  # Set to False if you want to preserve specific operators
)

# Run the model with the test input and print output for test verification
output = model(input_tensor)
print("Input:", input_tensor)
print("Output:", output)
```

#### Approach 2: Constructing an ONNX Graph Directly

For some test cases, you may want to construct the ONNX graph directly using the ONNX Python API.
This is particularly useful when:

- You need precise control over operator attributes
- You're testing operators that are difficult to trigger through PyTorch models
- You want to test specific graph structures

Example (see `tests/gather/gather_1d_idx.py` for a complete example):

```python
import numpy as np
import onnx
from onnx import TensorProto, helper

# Create inputs
data = np.random.randn(5, 5, 5).astype(np.float32)
indices = np.array([0, 2, 4], dtype=np.int64)

# Create node
node = helper.make_node(
    "Gather",
    inputs=["data", "indices"],
    outputs=["output"],
    axis=1
)

# Create input tensors
data_tensor = helper.make_tensor_value_info("data", TensorProto.FLOAT, data.shape)
indices_tensor = helper.make_tensor_value_info("indices", TensorProto.INT64, indices.shape)

# Create output tensor
output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [5, 3, 5])

# Create graph and model
graph = helper.make_graph(
    [node],
    "gather-model",
    [data_tensor, indices_tensor],
    [output_tensor],
    initializer=[]
)

model = helper.make_model(graph)
onnx.save(model, "tests/my_new_op/my_new_op.onnx")

# For test verification, print input and expected output
print("Data:", data)
print("Indices:", indices)
print("Expected output:", np.take(data, indices, axis=1))
```

### 2. Add the Build Step

Update `build.rs` to include your new model.

### 3. Create a mod.rs Test File

Create a test module file in your operator directory:

```sh
touch tests/my_new_op/mod.rs
```

Implement the test for your operator in this file:

```rust
use super::test_record_type::TestRecordType;
use burn_import::onnx::OnnxModel;

#[test]
fn test_my_new_op() {
    let model = OnnxModel::read("tests/my_new_op/my_new_op.onnx").unwrap();
    let record = model.into_record::<TestRecordType>();
    // Implement test logic and assertions here
}
```

Your test will be automatically included in the main test suite through `tests/test_mod.rs`.

## Best Practices for ONNX Testing

### Model Generation

1. **Keep Models Simple**: Focus on testing a single operator or a small group of related operators.

2. **Control Randomness**: Use fixed seeds in your Python scripts to ensure reproducible results:

   ```python
   torch.manual_seed(42)
   ```

3. **Print Test Values**: Always print your input and output tensors in the Python script for
   reference.

4. **Verify Operators**: Use [Netron](https://github.com/lutzroeder/netron) to verify your ONNX
   model contains the expected operators.

5. **Handle Constant Folding**: If PyTorch is optimizing away your operators, use:
   ```python
   torch.onnx.export(..., do_constant_folding=False)
   ```

### Test Implementation

1. **Test Multiple Cases**: Include tests for different input shapes, data types, and parameter
   combinations.

2. **Edge Cases**: Test edge cases like empty tensors, single-element tensors, or very large
   tensors.

3. **Parameter Variations**: If your operator has configurable parameters, test different parameter
   values.

4. **Numerical Precision**: Use appropriate tolerance levels based on operation sensitivity.

5. **Error Cases**: Test that invalid inputs are properly handled and appropriate errors are raised.

## Running Tests

### Default Backend

Run all tests with:

```sh
cargo test
```

This command runs all tests using the default backend: `burn::backend::NdArray<f32>`.

### Testing with Different Backends

You can test with different Burn backends by using feature flags:

#### WGPU Backend

```sh
cargo test --features test-wgpu
```

Uses `burn::backend::Wgpu` for GPU-accelerated computation.

#### LibTorch Backend

```sh
cargo test --features test-tch
```

Uses `burn::backend::LibTorch<f32>` for Torch backend integration.

#### NdArray Backend (Explicit)

```sh
cargo test --features test-ndarray
```

Explicitly uses `burn::backend::NdArray<f32>` (same as default).

### Running Specific Tests

Run tests for a specific operator:

```sh
cargo test --test test_mod my_new_op::test_my_new_op
```

Run a specific test with a selected backend:

```sh
cargo test --test test_mod my_new_op::test_my_new_op --features test-wgpu
```

### Supported Backends

- `burn::backend::NdArray<f32>` (default) - CPU-based computation using ndarray
- `burn::backend::Wgpu` - GPU-accelerated computation using WebGPU
- `burn::backend::LibTorch<f32>` - Torch backend integration

**Note:** Only one backend feature should be enabled at a time. The backend selection uses
conditional compilation with the following priority:

1. `test-wgpu` (highest priority)
2. `test-tch`
3. `test-ndarray` (default when no other backend is selected)

## Debugging Failed Tests

If a test fails, you can:

1. **Inspect ONNX Model**: Use Netron to visualize the model structure.

2. **Check Intermediate Values**: Add print statements in your Python script to see intermediate
   tensor values.

3. **Generate Rust Code**: Use the `burn-import` CLI to generate Rust code and inspect it:

   ```sh
   cargo run -p burn-import -- tests/my_new_op/my_new_op.onnx ./out
   ```

4. **Trace Through Conversion**: Add debug logging in your implementation to see where things might
   be going wrong.

5. **Numerical Issues**: If values are close but not equal, it might be a numerical precision issue.
   Try adjusting tolerance.
