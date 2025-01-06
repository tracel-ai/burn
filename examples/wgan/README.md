## Wasserstein Generative Adversarial Network
A burn implementation of examplar WGAN model to generate MNIST digits inspired by [the PyTorch implementation](https://bytepawn.com/training-a-pytorch-wasserstain-mnist-gan-on-google-colab.html). Please note that that better performance maybe gained by adopting a convolution layer in [some other models](https://github.com/Lornatang/WassersteinGAN-PyTorch).

### Usage:
For the MNIST dataset, each image has a size of 28x28 pixels and one color channel (grayscale), hence we use `--image-size 28 --channels 1` here.
#### Training
* cargo run --release --features ndarray -- train --artifact-dir output --num-epochs 200 --batch-size 64 --num-workers 10 --lr 0.0001 --latent-dim 100 --image-size 28 --channels 1 --num-critic 5 --clip-value 0.01 --sample-interval 1000
#### Generating
* cargo run --release --features ndarray -- generate --artifact-dir output
