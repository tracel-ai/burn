use burn_backend::{Scalar, ops::ActivationOps, tensor::FloatTensor};
use burn_backend_extension::backend_dispatch;

use crate::Dispatch;

#[backend_dispatch]
impl ActivationOps<Self> for Dispatch {
    fn leaky_relu(tensor: FloatTensor<Self>, negative_slope: Scalar) -> FloatTensor<Self> {
        B::leaky_relu(tensor, negative_slope)
    }

    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::relu(tensor)
    }

    fn relu_backward(output: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        B::relu_backward(output, grad)
    }

    fn gelu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::gelu(tensor)
    }

    fn prelu(tensor: FloatTensor<Self>, alpha: FloatTensor<Self>) -> FloatTensor<Self> {
        B::prelu(tensor, alpha)
    }

    fn gelu_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        B::gelu_backward(x, grad)
    }

    fn sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::sigmoid(tensor)
    }

    fn sigmoid_backward(output: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        B::sigmoid_backward(output, grad)
    }

    fn hard_sigmoid(tensor: FloatTensor<Self>, alpha: Scalar, beta: Scalar) -> FloatTensor<Self> {
        B::hard_sigmoid(tensor, alpha, beta)
    }

    fn softmax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        B::softmax(tensor, dim)
    }

    fn log_softmax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        B::log_softmax(tensor, dim)
    }

    fn softmin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        B::softmin(tensor, dim)
    }

    fn log_sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::log_sigmoid(tensor)
    }

    fn log_sigmoid_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        B::log_sigmoid_backward(x, grad)
    }
}
