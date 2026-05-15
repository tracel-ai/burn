mod backward;
mod forward;
mod kernel;

use burn::{
    backend::{Autodiff, Dispatch, Wgpu, backend_extension, tensor::FloatTensor},
    tensor::{Tensor, activation, kind::BridgeTensor},
};

/// We create our own Backend trait that extends the Burn backend trait.
#[backend_extension(Autodiff, Wgpu)]
pub trait Backend: burn::backend::Backend {
    fn fused_matmul_add_relu(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

/// We create our own AutodiffBackend trait that extends the Burn autodiff backend trait.
pub trait AutodiffBackend: Backend + burn::backend::AutodiffBackend {}

/// We define our custom implementation using the added function on our custom backend.
pub fn matmul_add_relu_custom(lhs: Tensor<3>, rhs: Tensor<3>, bias: Tensor<3>) -> Tensor<3> {
    let output = Dispatch::fused_matmul_add_relu(
        lhs.into_primitive().into(),
        rhs.into_primitive().into(),
        bias.into_primitive().into(),
    );

    Tensor::from_primitive(BridgeTensor::Float(output))
}

/// We define a reference implementation using basic tensor operations.
pub fn matmul_add_relu_reference(lhs: Tensor<3>, rhs: Tensor<3>, bias: Tensor<3>) -> Tensor<3> {
    let x = lhs.matmul(rhs) + bias;

    activation::relu(x)
}
