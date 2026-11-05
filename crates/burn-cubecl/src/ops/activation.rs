use crate::{
    CubeBackend,
    kernel::{
        GeluBackwardOp, SigmoidBackwardOp, launch_binop_float,
        unary_basic::{self, BasicFloatUnaryKind},
    },
};
use burn_backend::{ops::ActivationOps, tensor::FloatTensor};

impl ActivationOps<Self> for CubeBackend {
    fn gelu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch(tensor, |_| BasicFloatUnaryKind::Gelu)
    }

    fn gelu_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        launch_binop_float::<GeluBackwardOp>(x, grad)
    }

    fn sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch(tensor, |_| BasicFloatUnaryKind::Sigmoid)
    }

    fn sigmoid_backward(output: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        launch_binop_float::<SigmoidBackwardOp>(output, grad)
    }
}
