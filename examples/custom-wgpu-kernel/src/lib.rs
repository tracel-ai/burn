mod backward;
mod forward;

use burn::{
    backend::{Dispatch, backend_extension, tensor::FloatTensor},
    tensor::{Shape, Tensor, activation},
};

/// We create our own Backend trait that extends the Burn backend trait.
#[backend_extension(Autodiff, Cube, Fusion)]
pub trait Backend: burn::backend::Backend {
    #[fusion(dtype = lhs, shape = output_shape(lhs, rhs, bias))]
    fn fused_matmul_add_relu(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        bias: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

fn output_shape(lhs: &Shape, rhs: &Shape, bias: &Shape) -> Shape {
    let rank = lhs.num_dims();
    assert!(rank >= 2, "matmul needs at least two dimensions");
    assert_eq!(rank, rhs.num_dims(), "matrix ranks must match");
    assert_eq!(
        lhs[rank - 1],
        rhs[rank - 2],
        "contraction dimensions must match"
    );
    let mut shape = lhs.clone();
    for i in 0..rank - 2 {
        assert!(
            lhs[i] == rhs[i] || lhs[i] == 1 || rhs[i] == 1,
            "batch dimensions must broadcast"
        );
        shape[i] = if lhs[i] == 1 { rhs[i] } else { lhs[i] };
    }
    shape[rank - 1] = rhs[rank - 1];
    assert_eq!(
        &shape, bias,
        "kernel requires bias to match the output shape"
    );
    shape
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
