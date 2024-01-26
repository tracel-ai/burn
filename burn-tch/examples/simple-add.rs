use burn_tch::{LibTorch, LibTorchDevice};
use burn_tensor::Tensor;

fn main() {
    // let device = match tch::utils::has_cuda() {
    //     true => LibTorchDevice::Cuda(0),
    //     false => match tch::utils::has_mps() {
    //         true => LibTorchDevice::Mps,
    //         false => LibTorchDevice::Cpu,
    //     },
    // };
    let device = LibTorchDevice::Cpu;

    // Creation of two tensors, the first with explicit values and the second one with ones, with the same shape as the first
    let tensor_1 = Tensor::<LibTorch<f32>, 2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor_2 = Tensor::ones_like(&tensor_1);

    // Print the element-wise addition (done with the WGPU backend) of the two tensors.
    println!("{}", tensor_1 + tensor_2);
}
