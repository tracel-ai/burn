# Custom Image Dataset

The [custom-image-dataset](src/dataset.rs) example leverages the `ImageFolderDataset` to retrieve
dataset elements from a folder structure on disk. For this example, we use the
[CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). Since the original source is in
binary format, the data is downloaded from a
[fastai mirror](https://github.com/fastai/fastai/blob/master/fastai/data/external.py#L44) in a
folder structure with `.png` images to illustrate the `ImageFolderDataset` usage.

## Example Usage

```sh
cargo run --example custom-image-dataset
```
