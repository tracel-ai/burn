use crate::{BackendRouter, RouterChannel, RouterClient};
use burn_backend::{Scalar, ops::ActivationOps, tensor::FloatTensor};
use burn_ir::{
    ActivationOperationIr, BinaryOpIr, DimOpIr, HardSigmoidOpIr, OperationIr, OperationOutput,
    ScalarOpIr, UnaryOpIr,
};

impl<R: RouterChannel> ActivationOps<Self> for BackendRouter<R> {
    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Activation(ActivationOperationIr::Relu(desc)))
            .output()
    }

    fn relu_backward(output: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = output.client.clone();
        let desc = BinaryOpIr::create(output.into_ir(), grad.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Activation(
                ActivationOperationIr::ReluBackward(desc),
            ))
            .output()
    }

    fn leaky_relu(tensor: FloatTensor<Self>, negative_slope: Scalar) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = ScalarOpIr::create(tensor.into_ir(), negative_slope.into(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Activation(ActivationOperationIr::LeakyRelu(
                desc,
            )))
            .output()
    }

    fn prelu(tensor: FloatTensor<Self>, alpha: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = BinaryOpIr::create(tensor.into_ir(), alpha.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Activation(ActivationOperationIr::PRelu(desc)))
            .output()
    }

    fn gelu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Activation(ActivationOperationIr::Gelu(desc)))
            .output()
    }

    fn gelu_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc = BinaryOpIr::create(x.into_ir(), grad.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Activation(
                ActivationOperationIr::GeluBackward(desc),
            ))
            .output()
    }

    fn sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Activation(ActivationOperationIr::Sigmoid(
                desc,
            )))
            .output()
    }

    fn sigmoid_backward(output: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = output.client.clone();
        let desc = BinaryOpIr::create(output.into_ir(), grad.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Activation(
                ActivationOperationIr::SigmoidBackward(desc),
            ))
            .output()
    }

    fn hard_sigmoid(tensor: FloatTensor<Self>, alpha: Scalar, beta: Scalar) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = HardSigmoidOpIr::create(tensor.into_ir(), alpha.into(), beta.into(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Activation(ActivationOperationIr::HardSigmoid(
                desc,
            )))
            .output()
    }

    fn log_sigmoid(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Activation(ActivationOperationIr::LogSigmoid(
                desc,
            )))
            .output()
    }

    fn log_sigmoid_backward(x: FloatTensor<Self>, grad: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = x.client.clone();
        let desc = BinaryOpIr::create(x.into_ir(), grad.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Activation(
                ActivationOperationIr::LogSigmoidBackward(desc),
            ))
            .output()
    }

    fn softmax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::Activation(ActivationOperationIr::Softmax(
                desc,
            )))
            .output()
    }

    fn log_softmax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::Activation(ActivationOperationIr::LogSoftmax(
                desc,
            )))
            .output()
    }

    fn softmin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::Activation(ActivationOperationIr::Softmin(
                desc,
            )))
            .output()
    }
}
