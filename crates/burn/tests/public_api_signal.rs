#![cfg(feature = "signal")]

use burn::tensor::Tensor;

#[test]
fn signal_compatibility_paths_compile() {
    let _: burn_signal::StftOptions = burn::signal::StftOptions::default();
    let _: burn::tensor::signal::StftOptions = burn::signal::StftOptions::default();
    let _rfft: fn(Tensor<1>, usize, Option<usize>) -> (Tensor<1>, Tensor<1>) = burn::signal::rfft;
    let _irfft: fn(Tensor<1>, Tensor<1>, usize, Option<usize>) -> Tensor<1> =
        burn::tensor::signal::irfft;
}
