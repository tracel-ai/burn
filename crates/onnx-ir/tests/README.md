# ONNX-IR Integration Tests

This directory contains integration tests for the ONNX-IR parser infrastructure.

These tests verify the **conversion pipeline and infrastructure**, not specific node
implementations. They test how ONNX models are parsed, converted, and optimized through the 5-phase
pipeline.

## Adding New Infrastructure Tests

1. Create a Python script in `scripts/` (e.g., `scripts/generate_foo_model.py`):

   ```python
   #!/usr/bin/env -S uv run
   # /// script
   # dependencies = [
   #   "onnx>=1.15.0",
   #   "numpy>=1.24.0",
   # ]
   # ///

   # Save to: output_path = '../fixtures/foo.onnx'
   ```

2. Make it executable: `chmod +x scripts/generate_foo_model.py`

3. Run the script to generate the ONNX model: `cd scripts && ./generate_foo_model.py`

4. Add a Rust test in `basic.rs` or `infrastructure.rs`:

   ```rust
   mod test_utils;
   use test_utils::*;

   #[test]
   fn test_my_infrastructure_feature() {
       let graph = load_onnx("foo.onnx");

       // Test infrastructure features, not specific nodes
       // Focus on: value sources, type inference, graph structure, etc.
       assert_eq!(count_operation_nodes(&graph), 5);
   }
   ```

5. Run the tests to verify: `cargo test --package onnx-ir --test infrastructure`

## Test Philosophy

**DO test**:

- Pipeline phases working correctly
- Value source tracking (Static/Constant/Dynamic)
- Type inference convergence
- Graph structure handling
- Constant elimination
- Data flow through phases
- Infrastructure invariants

**DON'T test**:

- Specific node operator implementations (Conv2d, Gemm, etc.)
- Detailed node configs (those are covered in unit tests)
- Code generation (that's burn-onnx's job)

Focus on **how the pipeline works**, not **what operations it supports**.

## Requirements

- **uv**: For running Python scripts with automatic dependency management
  - Install: `pip install uv` or `cargo install uv`
- **Rust**: For running the smoke tests
- **Python 3.8+**: Required by uv (installed automatically with dependencies)
