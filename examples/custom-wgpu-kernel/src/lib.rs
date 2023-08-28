mod backward;
mod forward;
use burn::tensor::{
    activation,
    backend::{ADBackend, Backend},
    Tensor,
};

pub trait CustomBackend: Backend {
    fn fused_matmul_add_relu<const D: usize>(
        lhs: <Self as Backend>::TensorPrimitive<D>,
        rhs: <Self as Backend>::TensorPrimitive<D>,
        bias: <Self as Backend>::TensorPrimitive<D>,
    ) -> <Self as Backend>::TensorPrimitive<D>;
}

pub trait CustomADBackend: CustomBackend + ADBackend {}

pub fn matmul_add_relu_reference<B: Backend>(
    lhs: Tensor<B, 3>,
    rhs: Tensor<B, 3>,
    bias: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let x = lhs.matmul(rhs) + bias;

    activation::relu(x)
}

pub fn matmul_add_relu_custom<B: CustomBackend>(
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
