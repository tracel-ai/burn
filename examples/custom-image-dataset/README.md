# Training on a Custom Image Dataset

In this example, a [simple CNN](src/model.rs) model is trained from scratch on the
[CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) by leveraging the
`ImageFolderDataset` struct to retrieve images from a folder structure on disk.

Since the original source is in binary format, the data is downloaded from a
[fastai mirror](https://github.com/fastai/fastai/blob/master/fastai/data/external.py#L44) in a
folder structure with `.png` images.

```
cifar10
├── labels.txt
├── test
│   ├── airplane
│   ├── automobile
│   ├── bird
│   ├── cat
│   ├── deer
│   ├── dog
│   ├── frog
│   ├── horse
│   ├── ship
│   └── truck
└── train
    ├── airplane
    ├── automobile
    ├── bird
    ├── cat
    ├── deer
    ├── dog
    ├── frog
    ├── horse
    ├── ship
    └── truck
```

To load the training and test dataset splits, it is as simple as providing the root path to both
folders

```rust
let train_ds = ImageFolderDataset::new_classification("/path/to/cifar10/train").unwrap();
let test_ds = ImageFolderDataset::new_classification("/path/to/cifar10/test").unwrap();
```

as is done in [`CIFAR10Loader`](src/dataset.rs) for this example.

## Example Usage

The CNN model and training recipe used in this example are fairly simple since the objective is to
demonstrate how to load a custom image classification dataset from disk. Nonetheless, it still
achieves 70-80% accuracy on the test set after just 30 epochs.

Run it with the Torch GPU backend:

```sh
export TORCH_CUDA_VERSION=cu124
cargo run --example custom-image-dataset --release --features tch-gpu
```

Run it with our WGPU backend:

```sh
cargo run --example custom-image-dataset --release --features wgpu
```

Run it with our Metal backend:

```sh
cargo run --example custom-image-dataset --release --features metal
```
