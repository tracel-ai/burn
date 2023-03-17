use burn::tensor::backend::{ADBackend, Backend};
use burn::tensor::{Distribution, Tensor};
use burn_autodiff::ADBackendDecorator;
use burn_ndarray::NdArrayBackend;

fn f<B: Backend>(x: Tensor<B, 2>, weight: Tensor<B, 2>, bias: Tensor<B, 2>) -> Tensor<B, 2> {
    x.matmul(weight) + bias
}

fn df<B: ADBackend>(
    x: Tensor<B::InnerBackend, 2>,
    weight: Tensor<B::InnerBackend, 2>,
    bias: Tensor<B::InnerBackend, 2>,
) -> (Tensor<B::InnerBackend, 2>, Tensor<B::InnerBackend, 2>) {
    // Set the parameters
    let weight = Tensor::from_inner(weight).require_grad();
    let bias = Tensor::from_inner(bias).require_grad();

    // Call the forward pass
    let y = f::<B>(Tensor::from_inner(x), weight.clone(), bias.clone());

    // Call backward
    let grads = y.backward();

    // Fetch gradients for each parameter
    let grad_weight = weight.grad(&grads).unwrap();
    let grad_bias = bias.grad(&grads).unwrap();

    // Return the gradients
    (grad_weight, grad_bias)
}

fn main() {
    type Backend = NdArrayBackend<f32>;

    let weight = Tensor::random([3, 3], Distribution::Standard);
    let bias = Tensor::zeros([1, 3]);
    let x = Tensor::random([3, 3], Distribution::Standard);

    let y = f::<Backend>(x.clone(), weight.clone(), bias.clone()); // Compiles

    // let _grads = d_linear::<B>(x, weight, bias); // Doesn't compile
    let (grad_w, grad_b) = df::<ADBackendDecorator<Backend>>(x, weight, bias); // Compiles

    println!("--- Output ---\n{:?}\n", y);
    println!(
        "--- Gradients ---\nWeight: {:?}\n\nBias: {:?}",
        grad_w, grad_b
    );
}
