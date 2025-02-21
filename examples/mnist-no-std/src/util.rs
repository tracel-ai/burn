use crate::proto::{MnistImage, MNIST_IMAGE_HEIGHT, MNIST_IMAGE_WIDTH};

use burn::prelude::*;

// Convert an image into Tensor
// Originally copy from burn/examples/mnist-inference-web
pub fn image_to_tensor<B: Backend>(device: &B::Device, image: &MnistImage) -> Tensor<B, 3> {
    let tensor = TensorData::from(image.as_slice()).convert::<B::FloatElem>();
    let tensor = Tensor::<B, 1>::from_data(tensor, device);
    let tensor = tensor.reshape([1, MNIST_IMAGE_WIDTH, MNIST_IMAGE_HEIGHT]);

    // Normalize input: make between [0,1] and make the mean=0 and std=1
    // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
    // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122
    ((tensor / 255) - 0.1307) / 0.3081
}

pub fn images_to_tensors<B: Backend>(device: &B::Device, images: &[MnistImage]) -> Tensor<B, 3> {
    let tensors = images.iter().map(|v| image_to_tensor(device, v)).collect();
    Tensor::cat(tensors, 0)
}

pub fn labels_to_tensors<B: Backend>(device: &B::Device, labels: &[u8]) -> Tensor<B, 1, Int> {
    let targets = labels
        .iter()
        .map(|item| Tensor::<B, 1, Int>::from_data([(*item as i64).elem::<B::IntElem>()], device))
        .collect();
    Tensor::cat(targets, 0)
}
