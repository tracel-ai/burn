//! The data gradient of a convolution, which routes through the transposed-convolution
//! fallback in `burn-cubecl`.
//!
//! Only the input is tracked in each case, so the gradient that comes back is that path's
//! output alone, with no weight gradient mixed in.

use super::*;
use burn_tensor::{Device, Distribution, Shape, Tolerance, module, ops::ConvOptions};

/// `D` is the tensor rank and `N` its count of spatial dimensions, so `D == N + 2`.
#[track_caller]
fn assert_dgrad_matches_reference<const D: usize, const N: usize>(
    x_shape: [usize; D],
    w_shape: [usize; D],
    options: ConvOptions<N>,
    conv: fn(Tensor<D>, Tensor<D>, Option<Tensor<1>>, ConvOptions<N>) -> Tensor<D>,
) {
    let device = Device::default().autodiff();
    let ref_device = ReferenceDevice::new().autodiff();

    device.seed(0);

    let x = Tensor::<D>::random(x_shape, Distribution::Default, &device);
    let w = Tensor::<D>::random(w_shape, Distribution::Default, &device);

    let x_ref = Tensor::<D>::from_data(x.to_data(), &ref_device);
    let w_ref = Tensor::<D>::from_data(w.to_data(), &ref_device);

    let x = x.require_grad();
    let x_ref = x_ref.require_grad();

    let out = conv(x.clone(), w, None, options.clone());

    // A random upstream gradient rather than the ones `sum()` alone would send back: under a
    // uniform gradient, an index mistake that only permutes terms within the reduction cancels.
    let upstream = Tensor::<D>::random(out.shape(), Distribution::Default, &device);
    let upstream_ref = Tensor::<D>::from_data(upstream.to_data(), &ref_device);

    let grads = (out * upstream).sum().backward();
    let grads_ref = (conv(x_ref.clone(), w_ref, None, options) * upstream_ref)
        .sum()
        .backward();

    let grad = x.grad(&grads).expect("the input was tracked");
    let grad_ref = x_ref.grad(&grads_ref).expect("the input was tracked");

    assert_eq!(grad.shape(), Shape::from(x_shape));
    grad.into_data()
        .assert_approx_eq::<FloatElem>(&grad_ref.into_data(), Tolerance::default());
}

#[test]
fn conv1d_dgrad_should_match_reference_backend() {
    assert_dgrad_matches_reference(
        [4, 8, 17],
        [6, 8, 5],
        ConvOptions::new([1], [2], [1], 1),
        module::conv1d,
    );
}

#[test]
fn conv1d_dgrad_strided_dilated_grouped_should_match_reference_backend() {
    assert_dgrad_matches_reference(
        [4, 8, 17],
        [6, 4, 3],
        ConvOptions::new([2], [3], [2], 2),
        module::conv1d,
    );
}

#[test]
fn conv1d_dgrad_end_padding_should_match_reference_backend() {
    assert_dgrad_matches_reference(
        [2, 4, 9],
        [6, 2, 4],
        ConvOptions::new_with_padding([2], [(0, 3)], [2], 2),
        module::conv1d,
    );
}

#[test]
fn conv2d_dgrad_should_match_reference_backend() {
    assert_dgrad_matches_reference(
        [4, 8, 9, 11],
        [6, 8, 3, 5],
        ConvOptions::new([1, 1], [1, 2], [1, 1], 1),
        module::conv2d,
    );
}

/// Distinct sizes and options per axis, which a height-for-width mix-up would not survive.
#[test]
fn conv2d_dgrad_asymmetric_axes_should_match_reference_backend() {
    assert_dgrad_matches_reference(
        [4, 8, 9, 11],
        [6, 4, 2, 5],
        ConvOptions::new([3, 2], [2, 1], [1, 2], 2),
        module::conv2d,
    );
}

#[test]
fn conv2d_dgrad_end_padding_should_match_reference_backend() {
    assert_dgrad_matches_reference(
        [2, 4, 7, 8],
        [6, 2, 2, 3],
        ConvOptions::new_with_padding([1, 2], [(0, 1), (0, 2)], [1, 2], 2),
        module::conv2d,
    );
}

#[test]
fn conv2d_dgrad_depthwise_should_match_reference_backend() {
    assert_dgrad_matches_reference(
        [2, 6, 9, 11],
        [6, 1, 3, 3],
        ConvOptions::new([1, 1], [1, 1], [1, 1], 6),
        module::conv2d,
    );
}

/// 3D has no NHWC transposed-convolution kernel, so it still runs through NCHW.
#[test]
fn conv3d_dgrad_should_match_reference_backend() {
    assert_dgrad_matches_reference(
        [2, 4, 5, 6, 7],
        [6, 2, 3, 2, 3],
        ConvOptions::new([2, 1, 1], [1, 2, 0], [1, 1, 2], 2),
        module::conv3d,
    );
}
