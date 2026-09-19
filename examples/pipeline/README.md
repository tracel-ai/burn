# Pipeline

Runs a model split by whole layers across several devices. The model is one input projection, a
stack of identical blocks and an output head, which is the shape pipeline parallelism exists for:
the blocks are what gets shared out.

```
src/model.rs      the model, its config, and its `Pipeline` implementation
src/devices.rs    which devices to split across
src/inference.rs  place, load the weights onto each stage, run the split forward
src/training.rs   place on autodiff devices, then train with Adam
```

The order is the point. `ModelConfig::init` builds the layers lazily, so nothing is allocated;
`place` gives each parameter the device of the segment that owns it, still before any of them
exist; and `load_record` then puts each weight straight onto its own stage. The model never exists
whole on one device, which is what lets one too large for a single card load at all.

`pipeline-infer` stands an initialized model in for a trained one, where a real program would call
`ModuleRecord::load`, and checks the predictions against the same weights on one device.
`pipeline-train` places the model on autodiff devices instead and trains it with Adam, each
parameter updated on the stage it lives on. It runs on Flex alone without a GPU, since the CPU
backend forwards but cannot launch the backward matmul.

Only one device works at a time here, so what this runs today is a model too large for one device
rather than a faster one. Overlapping the stages takes a schedule that splits a batch into
microbatches, such as GPipe or 1F1B, which is not implemented yet.

The blocks are shared out evenly across one device per GPU, found with `Device::enumerate_physical`
so a card reachable through both CUDA and Vulkan is used once, through CUDA. Without a GPU feature it
splits across two CPU backends, Flex and CPU, which is a real split on a machine with no GPU:

```bash
cargo run --example pipeline-infer --release                          # Flex and CPU, no GPU needed
cargo run --example pipeline-infer --release --features cuda          # every NVIDIA GPU
cargo run --example pipeline-infer --release --features rocm          # every supported AMD GPU
cargo run --example pipeline-infer --release --features metal         # every Apple GPU
cargo run --example pipeline-infer --release --features vulkan        # every GPU, through Vulkan
cargo run --example pipeline-infer --release --features cuda,vulkan   # every GPU, NVIDIA ones through CUDA
cargo run --example pipeline-train --release                          # train, Flex alone without a GPU
```

A card several runtimes reach is taken through the first of CUDA, ROCm, Metal, Vulkan, WebGPU that
the build has and that reaches it, so a native runtime wins over a portable one. On a box with two
NVIDIA and two AMD cards, `--features cuda,vulkan` gives a four-way split with the NVIDIA halves on
CUDA. Note that a carry crossing between two runtimes stages through host memory, so a placement
that groups cards of one runtime together moves far less than one that alternates.
