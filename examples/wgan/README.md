# Wasserstein Generative Adversarial Network

A burn implementation of an example WGAN model to generate MNIST digits inspired by
[the PyTorch implementation](https://bytepawn.com/training-a-pytorch-wasserstain-mnist-gan-on-google-colab.html).
Please note that better performance maybe gained by adopting a convolution layer in
[some other models](https://github.com/Lornatang/WassersteinGAN-PyTorch).

## Usage

## Training

```sh
# Cuda backend
cargo run --example wgan-mnist --release --features cuda

# Wgpu backend
cargo run --example wgan-mnist --release --features wgpu

# Tch GPU backend
export TORCH_CUDA_VERSION=cu128 # Set the cuda version
cargo run --example wgan-mnist --release --features tch-gpu

# Tch CPU backend
cargo run --example wgan-mnist --release --features tch-cpu

# NdArray backend (CPU)
cargo run --example wgan-mnist --release --features ndarray                # f32 - single thread
cargo run --example wgan-mnist --release --features ndarray-blas-openblas  # f32 - blas with openblas
cargo run --example wgan-mnist --release --features ndarray-blas-netlib    # f32 - blas with netlib
```

### Generating

To generate a sample of images, you can use `wgan-generate`. The same feature flags are used to select a backend.

```sh
cargo run --example wgan-generate --release --features cuda
```
