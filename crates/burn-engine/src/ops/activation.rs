use burn_backend::{Scalar, ops::ActivationOps, tensor::FloatTensor};

use crate::Engine;
use crate::backends::*;
use crate::{binary_op, unary_op};

impl ActivationOps<Self> for Engine {
    fn leaky_relu(tensor: FloatTensor<Self>, negative_slope: Scalar) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::leaky_relu(tensor, negative_slope) => Float)
    }

    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::relu(tensor) => Float)
    }

    fn relu_backward(output: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_op!((output, float), (grad, float), |output, grad| B::relu_backward(output, grad) => Float)
    }

    fn gelu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::gelu(tensor) => Float)
    }

    fn prelu(tensor: FloatTensor<Self>, alpha: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_op!((tensor, float), (alpha, float), |tensor, alpha| B::prelu(tensor, alpha) => Float)
    }

    fn gelu_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_op!((x, float), (grad, float), |x, grad| B::gelu_backward(x, grad) => Float)
    }

    fn sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::sigmoid(tensor) => Float)
    }

    fn sigmoid_backward(output: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_op!((output, float), (grad, float), |output, grad| B::sigmoid_backward(output, grad) => Float)
    }

    fn hard_sigmoid(tensor: FloatTensor<Self>, alpha: Scalar, beta: Scalar) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::hard_sigmoid(tensor, alpha, beta) => Float)
    }

    fn log_sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(tensor, float, |tensor| B::log_sigmoid(tensor) => Float)
    }

    fn log_sigmoid_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_op!((x, float), (grad, float), |x, grad| B::log_sigmoid_backward(x, grad) => Float)
    }
}
