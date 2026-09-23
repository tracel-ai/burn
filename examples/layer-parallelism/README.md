# Layer Parallelism

Runs a model split by whole layers across several devices. The model is an input projection, a
stack of identical blocks, a final norm and an output projection, and the blocks are what gets
shared out.

The model comes in two shapes. `Model` is how it trains on one device. `LayeredModel` is the same
model shaped to be split: an embedding layer, the blocks, and a head holding the final norm and the
output projection. The split shape is the example's own, and so is the mapping between the two: a
checkpoint of `Model` loads into `LayeredModel` by renaming its keys.

```
src/model.rs      the single-device model, its config, and the block both shapes share
src/layered.rs    the split shape, its layers, and the key mapping from a single-device checkpoint
src/devices.rs    which devices to split across
src/inference.rs  split, load the single-device checkpoint onto each device, run the split forward
src/training.rs   split across autodiff devices, then train with Adam
```

The order is the point. `LayeredModel::new` builds each layer on its own device, lazily, so
nothing is allocated; `DistributedLayeredModel::new` checks each layer is where the placement puts
it; and `load_from` then puts each weight straight onto its own device. The model never exists whole
on one device, which is what lets one too large for a single card load at all.

`layer-parallelism-infer` stands an initialized `Model` in for a trained one, where a real program
would read its checkpoint from a file, and checks the split predictions against `Model` itself.
`layer-parallelism-train` splits the model across autodiff devices instead and trains it with Adam,
each parameter updated on the device it lives on. It runs on Flex alone without a GPU, since the CPU
backend forwards but cannot launch the backward matmul.

Only one device works at a time here, so what this runs today is a model too large for one device
rather than a faster one. Overlapping the devices takes a schedule that splits a batch into
microbatches, such as GPipe or 1F1B, which is not implemented yet.

The blocks are shared out evenly across one device per GPU, found with `Device::enumerate_physical`
so a card reachable through both CUDA and Vulkan is used once, through CUDA. Without a GPU feature it
splits across two CPU backends, Flex and CPU, which is a real split on a machine with no GPU:

```bash
cargo run --example layer-parallelism-infer --release                          # Flex and CPU, no GPU needed
cargo run --example layer-parallelism-infer --release --features cuda          # every NVIDIA GPU
cargo run --example layer-parallelism-infer --release --features rocm          # every supported AMD GPU
cargo run --example layer-parallelism-infer --release --features metal         # every Apple GPU
cargo run --example layer-parallelism-infer --release --features vulkan        # every GPU, through Vulkan
cargo run --example layer-parallelism-infer --release --features cuda,vulkan   # every GPU, NVIDIA ones through CUDA
cargo run --example layer-parallelism-train --release                          # train, Flex alone without a GPU
```

A card several runtimes reach is taken through the first of CUDA, ROCm, Metal, Vulkan, WebGPU that
the build has and that reaches it, so a native runtime wins over a portable one. On a box with two
NVIDIA and two AMD cards, `--features cuda,vulkan` gives a four-way split with the NVIDIA halves on
CUDA. Note that a signal crossing between two runtimes stages through host memory, so a placement
that groups cards of one runtime together moves far less than one that alternates.
