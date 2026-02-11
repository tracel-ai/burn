use burn_backend::{Scalar, ops::ActivationOps, tensor::FloatTensor};

use crate::backends::*;
use crate::{Engine, EngineTensor};
use crate::{binary_float, unary_float};

impl ActivationOps<Self> for Engine {
    fn leaky_relu(tensor: FloatTensor<Self>, negative_slope: Scalar) -> FloatTensor<Self> {
        unary_float!(leaky_relu, tensor, negative_slope)
    }

    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(relu, tensor)
    }

    fn relu_backward(output: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!(relu_backward, output, grad)
    }

    fn gelu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(gelu, tensor)
    }

    fn prelu(tensor: FloatTensor<Self>, alpha: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!(prelu, tensor, alpha)
    }

    fn gelu_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!(gelu_backward, x, grad)
    }

    fn sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(sigmoid, tensor)
    }

    fn sigmoid_backward(output: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!(sigmoid_backward, output, grad)
    }

    fn hard_sigmoid(tensor: FloatTensor<Self>, alpha: Scalar, beta: Scalar) -> FloatTensor<Self> {
        unary_float!(hard_sigmoid, tensor, alpha, beta)
    }

    fn log_sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float!(log_sigmoid, tensor)
    }

    fn log_sigmoid_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float!(log_sigmoid_backward, x, grad)
    }
}
