use burn_backend::{TensorMetadata, ops::FloatTensorOps};
use burn_tch::{LibTorch, LibTorchDevice};

fn main() {
    type B = LibTorch<f32>;
    let device = LibTorchDevice::Cpu;

    // Creation of two tensors, the first with explicit values and the second one with ones, with the same shape as the first
    let tensor_1 = B::float_from_data([[2f32, 3.], [4., 5.]].into(), &device);
    let tensor_2 = B::float_ones(tensor_1.shape(), &device, tensor_1.dtype().into());

    // Print the element-wise addition of the two tensors.
    println!("{}", B::float_add(tensor_1, tensor_2));
}
