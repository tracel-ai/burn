#![cfg(feature = "linalg")]

use burn::tensor::Tensor;

#[test]
fn linalg_and_initializer_compatibility_paths_compile() {
    let _: burn_linalg::Norm = burn_linalg::Norm::L2;
    let _: burn::linalg::Norm = burn::linalg::Norm::L2;
    let _: burn::tensor::linalg::Norm = burn::tensor::linalg::Norm::L2;
    let _: burn::module::Initializer = burn::module::Initializer::Zeros;
    let _: burn::nn::Initializer = burn::nn::Initializer::Zeros;
    let _: burn_nn::Initializer = burn_nn::Initializer::Zeros;

    let _qr: fn(Tensor<2>, bool) -> (Tensor<2>, Tensor<2>) = burn::linalg::qr::<2>;
}
