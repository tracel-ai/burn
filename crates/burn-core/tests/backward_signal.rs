//! A derived `BackwardSignal` cuts, reads and weighs its float tensors and carries the rest.
//!
//! Run with `cargo test -p burn-core --features flex,autodiff --test backward_signal`.
#![cfg(all(feature = "flex", feature = "autodiff"))]

use burn_core as burn;

use burn::module::parallel::BackwardSignal;
use burn::tensor::{Bool, Device, Distribution, Tensor, Tolerance};

#[derive(BackwardSignal, Clone, Debug)]
struct Encoded {
    hidden: Tensor<2>,
    mask: Tensor<2, Bool>,
    #[backward_signal(skip)]
    length: usize,
}

#[derive(BackwardSignal, Clone, Debug)]
struct Pair(Tensor<2>, Option<Tensor<2>>);

fn device() -> Device {
    Device::flex().autodiff()
}

fn encoded(device: &Device) -> Encoded {
    Encoded {
        hidden: Tensor::random([2, 3], Distribution::Default, device),
        mask: Tensor::<2>::random([2, 3], Distribution::Default, device).greater_elem(0.5),
        length: 3,
    }
}

#[test]
fn a_cut_signal_tracks_its_float_tensors_and_carries_the_rest() {
    let device = device();
    let signal = encoded(&device);
    let mask = signal.mask.clone();

    let cut = signal.detach_require_grad();
    let grads = (cut.hidden.clone() * 2.0).sum().backward();
    let grad = cut.grad(&grads);

    grad.hidden.into_data().assert_approx_eq::<f32>(
        &Tensor::<2>::full([2, 3], 2.0, &device).into_data(),
        Tolerance::default(),
    );
    assert_eq!(grad.mask.into_data(), mask.into_data());
    assert_eq!(grad.length, 3);
}

#[test]
fn a_signal_weighs_each_float_tensor_by_its_gradient() {
    let device = device();
    let ones = || Tensor::<2>::ones([2, 3], &device);
    let pair = Pair(ones() * 2.0, Some(ones() * 3.0));
    let grad = Pair(ones() * 5.0, Some(ones() * 7.0));

    let product = pair
        .grad_product(grad)
        .expect("the pair holds float tensors");

    // 6 elements of 2 * 5 and 6 of 3 * 7.
    product.into_data().assert_approx_eq::<f32>(
        &Tensor::<1>::from_floats([186.0], &device).into_data(),
        Tolerance::default(),
    );
}

#[test]
fn a_signal_with_no_float_tensor_weighs_nothing() {
    let device = device();
    let mask = encoded(&device).mask;

    assert!(mask.clone().grad_product(mask).is_none());
}
