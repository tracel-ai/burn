mod backward;
mod forward;

use burn::tensor::{activation, Tensor};

/// We use a type alias for better readability.
pub type FloatTensor<B, const D: usize> = <B as burn::tensor::backend::Backend>::TensorPrimitive<D>;

/// We create our own Backend trait that extends the Burn backend trait.
pub trait Backend: burn::tensor::backend::Backend {
    fn fused_matmul_add_relu<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
        bias: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D>;
}

/// We create our own AutodiffBackend trait that extends the Burn autodiff backend trait.
pub trait AutodiffBackend: Backend + burn::tensor::backend::AutodiffBackend {}

/// We define our custom implementation using the added function on our custom backend.
pub fn matmul_add_relu_custom<B: Backend>(
    lhs: Tensor<B, 3>,
    rhs: Tensor<B, 3>,
    bias: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let output = B::fused_matmul_add_relu(
        lhs.into_primitive(),
        rhs.into_primitive(),
        bias.into_primitive(),
    );

    Tensor::from_primitive(output)
}

/// We define a reference implementation using basic tensor operations.
pub fn matmul_add_relu_reference<B: Backend>(
    lhs: Tensor<B, 3>,
    rhs: Tensor<B, 3>,
    bias: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let x = lhs.matmul(rhs) + bias;

    activation::relu(x)
}
