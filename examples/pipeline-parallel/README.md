# Pipeline Parallelism

Trains a small regression model with pipeline parallelism: the model is split by layers across
several devices. It implements `Pipeline` (an input projection, a stack of residual blocks, an
output head); `place` forks each segment's parameters onto its device, and `forward_on` moves the
activations between devices during training.

The blocks are shared out evenly across one device per GPU, found with `Device::enumerate_physical`
so a card reachable through both CUDA and Vulkan is used once, through CUDA. Without a GPU feature the
example splits across the CPU twice, so it runs anywhere:

```bash
cargo run --example pipeline-parallel --release                            # Flex, one device standing in for two
cargo run --example pipeline-parallel --release --features cuda            # every NVIDIA GPU
cargo run --example pipeline-parallel --release --features vulkan          # every GPU, through Vulkan
cargo run --example pipeline-parallel --release --features cuda,vulkan     # every GPU, NVIDIA ones through CUDA
```
