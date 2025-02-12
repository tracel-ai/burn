# Modified UNet
A burn implementation of a modified form of UNet (convolutional neural network) for image segmentation. This work is based on the original paper _U-Net: Convolutional Networks for Biomedical Image Segmentation_^1^ [arxiv](https://arxiv.org/abs/1505.04597).

## Model Architecture Implementation
### Key Components
The implementation consists of several key components:

1. **DoubleConv Block**
* A basic building block that performs two consecutive convolutions
* Each convolution is followed by batch normalization and ReLU activation
* Used throughout the network for feature extraction

2. **Down Block**
* Handles downsampling in the encoder path
* Combines max pooling with a DoubleConv block
* Reduces spatial dimensions while increasing feature channels

3. **Up Block**
* Handles upsampling in the decoder path
* Uses transposed convolution followed by a DoubleConv block
* Increases spatial dimensions while decreasing feature channels

4. **OutConv Block**
* Final output layer
* Single convolution with ReLU activation
* Maps features to the desired number of output channels

5. **UNet Structure** The main UNet struct combines these components in a classic U-shaped architecture:
* Encoder path: Initial DoubleConv followed by 4 Down blocks
* Decoder path: 4 Up blocks with skip connections
* Final output layer: OutConv block

### Encoder Path
The encoder path consists of a series of `Down` modules that downscale the input image while increasing the number of channels. The `Down` modules apply max pooling with a stride of 2, followed by a convolutional block.

* **inc:** The initial convolutional block that takes the input image and produces a feature map with 64 channels.
* **down1, down2, down3, down4:** The subsequent `Down` modules that downscale the feature map while increasing the number of channels to 128, 256, 512, and 1024, respectively.

### Decoder Path
The decoder path consists of a series of Up modules that upsample the feature map while decreasing the number of channels. The `Up` modules apply transposed convolution with a stride of 2, followed by a convolutional block.

* **up4, up3, up2, up1:** The `Up` modules that upsample the feature map while decreasing the number of channels to 512, 256, 128, and 64, respectively.

### Final Output Layer
The final output layer consists of an `OutConv` module that applies a convolutional operation with a kernel size of 1x1 to produce the final output.

### Forward Pass
The forward pass through the U-Net architecture involves the following steps:

1. The input tensor `[batch_size, 3, height, width]` (3 channel images) is passed through the `inc` convolutional block to produce a feature map with 64 channels.
2. The feature map is then passed through the `down1`, `down2`, `down3`, and `down4` modules progressively reducing spatial dimensions while increasing channels to produce a feature map with 1024 channels.
3. The feature map is then passed through the `up4`, `up3`, `up2`, and `up1` modules progressively increasing spatial dimensions while decreasing channels to produce a feature map with 64 channels.
4. The feature map is then passed through the `outc` convolutional block to produce the final output tensor `[batch_size, 1, height, width]` (single channel segmentation mask).

### Skip Connections
The implementation also includes skip connections between the encoder and decoder paths. The skip connections allow the model to preserve spatial information and improve the accuracy of the segmentation.

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
