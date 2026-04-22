# MNIST

The example is showing you how to:

- Define your own custom module (MLP).
- Create the data pipeline from a raw dataset to a batched multi-threaded fast DataLoader.
- Configure a learner to display and log metrics as well as to keep training checkpoints.

The example can be run like so:

```bash
git clone https://github.com/tracel-ai/burn.git
cd burn
# Use the --release flag to really speed up training.
echo "Using flex backend"
cargo run --example mnist --release --features flex                   # CPU Flex Backend - f32
echo "Using tch backend"
export TORCH_CUDA_VERSION=cu128                                       # Set the cuda version
cargo run --example mnist --release --features tch-gpu                # GPU Tch Backend - f32
cargo run --example mnist --release --features tch-cpu                # CPU Tch Backend - f32
echo "Using vulkan backend"
cargo run --example mnist --release --features vulkan
```
