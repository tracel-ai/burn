# Import PyTorch Weights

This crate provides a simple example for importing PyTorch generated weights to Burn.

The `.pt` file is converted into a Burn consumable file (message pack format) using `burn-import`.
The conversation is done in the `build.rs` file.

The model is separated into a sub-crate because `build.rs` needs for conversion and build.rs cannot
import modules for the same crate.

## Usage

```bash
cargo run -- 15
```

Output:

```bash
Finished dev [unoptimized + debuginfo] target(s) in 0.13s
    Running `burn/target/debug/onnx-inference 15`

Image index: 15
Success!
Predicted: 5
Actual: 5
See the image online, click the link below:
https://huggingface.co/datasets/ylecun/mnist/viewer/mnist/test?row=15
```
