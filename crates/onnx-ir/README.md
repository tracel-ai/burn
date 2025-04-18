# ONNX-IR

ONNX-IR is a pure Rust library for parsing ONNX models into an intermediate representation that can be used to generate code for various ML/DL frameworks. It's part of the Burn project, with key features including ONNX model parsing, rank inference, and node remapping. The crate supports converting ONNX models to Burn graphs and includes utilities for handling constants and graph transformations.

For a full list of currently supported operators, please check [here](https://github.com/tracel-ai/burn/blob/main/crates/burn-import/SUPPORTED-ONNX-OPS.md)

To see how to use this for generating burn graphs, see [here](crates/burn-import/src/onnx/to_burn.rs).