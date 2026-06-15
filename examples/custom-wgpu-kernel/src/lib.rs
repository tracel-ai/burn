mod backward;
mod forward;

use burn::{
    backend::{Autodiff, Dispatch, Wgpu, backend_extension, tensor::FloatTensor},
    tensor::{Tensor, activation},
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

/// We define our custom implementation using the added function on our custom backend.
pub fn matmul_add_relu_custom(lhs: Tensor<3>, rhs: Tensor<3>, bias: Tensor<3>) -> Tensor<3> {
    let output = Dispatch::fused_matmul_add_relu(
        lhs.into_dispatch(),
        rhs.into_dispatch(),
        bias.into_dispatch(),
    );

    Tensor::from_dispatch(output)
}

/// We define a reference implementation using basic tensor operations.
pub fn matmul_add_relu_reference(lhs: Tensor<3>, rhs: Tensor<3>, bias: Tensor<3>) -> Tensor<3> {
    let x = lhs.matmul(rhs) + bias;

    activation::relu(x)
}
