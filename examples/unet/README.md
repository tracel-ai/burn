# Modified UNet
A burn implementation of a modified form of UNet (convolutional neural network) for image segmentation. This work is based on the original paper _U-Net: Convolutional Networks for Biomedical Image Segmentation_^1^ [arxiv](https://arxiv.org/abs/1505.04597). For more information about the model architecture in this implementation, see [MODEL ARCHITECTURE](./MODEL_ARCHITECTURE.md).

## Usage

> [!CAUTION]
> Data is not provided for this example and it will not run. See [DATA](./DATA.md) for an explanation of the requirements and a link to sample data publicly available.

### Training
```sh
cargo run --example unet_train --release
```

### Inference
```sh
cargo run --example unet_infer --release
```


---
References:

Ronneberger, O., Fischer, P., Brox, T., 2015. U-net: Convolutional networks for biomedical image segmentation, in: International Conference on Medical image computing and computer-assisted intervention, Springer. pp. 234–241.
