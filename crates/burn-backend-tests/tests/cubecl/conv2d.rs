use super::*;
use burn_tensor::Tolerance;
use burn_tensor::ops::ConvOptions;
use burn_tensor::{Distribution, module};

#[test]
fn conv2d_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let input = TestTensor::<4>::random([6, 16, 32, 32], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([12, 8, 3, 3], Distribution::Default, &device);
    let bias = TestTensor::<1>::random([12], Distribution::Default, &device);

    let input_ref = TestTensor::<4>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);
    let bias_ref = TestTensor::<1>::from_data(bias.to_data(), &ref_device);

    let options = ConvOptions::new([2, 3], [2, 3], [2, 3], 2);

    let output = module::conv2d(input, weight, Some(bias), options.clone());
    let output_ref = module::conv2d(input_ref, weight_ref, Some(bias_ref), options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

#[test]
fn conv2d_should_match_reference_backend_implicit() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let input = TestTensor::<4>::random([4, 16, 6, 6], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([16, 16, 3, 3], Distribution::Default, &device);
    let bias = TestTensor::<1>::random([16], Distribution::Default, &device);

    let input_ref = TestTensor::<4>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);
    let bias_ref = TestTensor::<1>::from_data(bias.to_data(), &ref_device);

    let options = ConvOptions::new([1, 1], [2, 2], [1, 1], 1);

    let output = module::conv2d(input, weight, Some(bias), options.clone());
    let output_ref = module::conv2d(input_ref, weight_ref, Some(bias_ref), options);

    let tolerance = Tolerance::default();
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), tolerance);
}

/// Regression test for bias loader in new implicit GEMM
#[test]
fn conv2d_should_match_reference_backend_bias_regression() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let input = TestTensor::<4>::random([1, 1, 1, 1], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([32, 1, 3, 3], Distribution::Default, &device);
    let bias = TestTensor::<1>::random([32], Distribution::Default, &device);

    let input_ref = TestTensor::<4>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);
    let bias_ref = TestTensor::<1>::from_data(bias.to_data(), &ref_device);

    let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);

    let output = module::conv2d(input, weight, Some(bias), options.clone()).permute([0, 2, 3, 1]);
    let output_ref =
        module::conv2d(input_ref, weight_ref, Some(bias_ref), options).permute([0, 2, 3, 1]);

    let tolerance = Tolerance::default();
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), tolerance);
}

#[test]
fn conv2d_weight_backward_should_run() {
    // https://github.com/tracel-ai/burn/issues/4226#issuecomment-3911335769
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
    let x = TestTensor::<4>::random([1, 1, 1, 672], Distribution::Default, &device);
    // let x = x.permute([0, 3, 1, 2]);

    let output_grad = TestTensor::<4>::random([1, 168, 1, 1], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([168, 672, 1, 1], Distribution::Default, &device);

    let x_ref = TestTensor::<4>::from_data(x.to_data(), &ref_device);
    let output_grad_ref = TestTensor::<4>::from_data(output_grad.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);

    // Input shape [672, 1] and strides [672, 672] should be valid
    let output = module::conv2d_weight_backward(
        x.permute([0, 3, 1, 2]),
        weight,
        output_grad,
        options.clone(),
    );

    // Input shape [672, 1] and strides [672, 672] should be valid
    let output_ref = module::conv2d_weight_backward(
        x_ref.permute([0, 3, 1, 2]),
        weight_ref,
        output_grad_ref,
        options,
    );

    let tolerance = Tolerance::default();
    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), tolerance);
}

/// A pointwise weight gradient is a matmul over every pixel in the batch, not a
/// convolution.
///
/// The channel counts differ from each other and are not powers of two, so a
/// pitched allocator pads the rows the matmul reads.
#[test]
fn conv2d_weight_backward_pointwise_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let x = TestTensor::<4>::random([3, 6, 5, 7], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([10, 6, 1, 1], Distribution::Default, &device);
    let output_grad = TestTensor::<4>::random([3, 10, 5, 7], Distribution::Default, &device);

    let x_ref = TestTensor::<4>::from_data(x.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);
    let output_grad_ref = TestTensor::<4>::from_data(output_grad.to_data(), &ref_device);

    let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);

    let output = module::conv2d_weight_backward(x, weight, output_grad, options.clone());
    let output_ref = module::conv2d_weight_backward(x_ref, weight_ref, output_grad_ref, options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

/// A pointwise weight gradient whose contraction is long enough to be cut.
///
/// The cut only applies above a threshold, so every other convolution test here
/// is below it and leaves that path unrun: `batch * height * width` has to
/// reach a few thousand before there is an imbalance worth correcting. This is
/// the smallest shape that reaches it and still divides evenly.
///
/// The channel counts are unequal and not powers of two, so the partial
/// gradients are summed over rows a pitched allocator has padded.
#[test]
fn conv2d_weight_backward_pointwise_split_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let x = TestTensor::<4>::random([4, 6, 64, 64], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([10, 6, 1, 1], Distribution::Default, &device);
    let output_grad = TestTensor::<4>::random([4, 10, 64, 64], Distribution::Default, &device);

    let x_ref = TestTensor::<4>::from_data(x.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);
    let output_grad_ref = TestTensor::<4>::from_data(output_grad.to_data(), &ref_device);

    let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);

    let output = module::conv2d_weight_backward(x, weight, output_grad, options.clone());
    let output_ref = module::conv2d_weight_backward(x_ref, weight_ref, output_grad_ref, options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

/// The same, with a contraction no even cut divides.
///
/// `batch * height * width` is `1 * 65 * 65`, which is odd — long enough to be
/// worth cutting and impossible to cut into equal pieces, since the whole point
/// of the cut is that the reshape splitting it is free. The path has to fall
/// back on the uncut form rather than round, and the gradient still has to come
/// out right. It falls back rather than declining because the autotune key
/// holds the spatial dimensions anchored, so this shape shares a key with ones
/// the cut does divide: an `Err` here would abort on their cached hit.
#[test]
fn conv2d_weight_backward_pointwise_odd_contraction_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let x = TestTensor::<4>::random([1, 6, 65, 65], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([10, 6, 1, 1], Distribution::Default, &device);
    let output_grad = TestTensor::<4>::random([1, 10, 65, 65], Distribution::Default, &device);

    let x_ref = TestTensor::<4>::from_data(x.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);
    let output_grad_ref = TestTensor::<4>::from_data(output_grad.to_data(), &ref_device);

    let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);

    let output = module::conv2d_weight_backward(x, weight, output_grad, options.clone());
    let output_ref = module::conv2d_weight_backward(x_ref, weight_ref, output_grad_ref, options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

/// A dense weight gradient long enough for the contraction to be cut.
///
/// The dense path lays the input out as columns and contracts them, so it has
/// to agree with the convolution-by-the-gradient form it replaces at a size
/// where the cut also applies — `batch * height * width` here is 16384, well
/// over the threshold, where every other convolution test is far under it.
#[test]
fn conv2d_weight_backward_dense_split_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let x = TestTensor::<4>::random([4, 6, 64, 64], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([10, 6, 3, 3], Distribution::Default, &device);
    let output_grad = TestTensor::<4>::random([4, 10, 64, 64], Distribution::Default, &device);

    let x_ref = TestTensor::<4>::from_data(x.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);
    let output_grad_ref = TestTensor::<4>::from_data(output_grad.to_data(), &ref_device);

    let options = ConvOptions::new([1, 1], [1, 1], [1, 1], 1);

    let output = module::conv2d_weight_backward(x, weight, output_grad, options.clone());
    let output_ref = module::conv2d_weight_backward(x_ref, weight_ref, output_grad_ref, options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

/// A dense weight gradient with no padding at all.
///
/// Every kernel tap then covers the whole output, so the columns are written
/// end to end and are allocated uninitialised rather than zeroed. Nothing else
/// here reaches that: the unpadded convolution tests are all grouped or
/// pointwise, which this path declines. The kernel is not square, so the two
/// spatial axes cannot pass by symmetry.
#[test]
fn conv2d_weight_backward_dense_unpadded_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let x = TestTensor::<4>::random([3, 5, 33, 35], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([7, 5, 3, 2], Distribution::Default, &device);
    let output_grad = TestTensor::<4>::random([3, 7, 31, 34], Distribution::Default, &device);

    let x_ref = TestTensor::<4>::from_data(x.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);
    let output_grad_ref = TestTensor::<4>::from_data(output_grad.to_data(), &ref_device);

    let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);

    let output = module::conv2d_weight_backward(x, weight, output_grad, options.clone());
    let output_ref = module::conv2d_weight_backward(x_ref, weight_ref, output_grad_ref, options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

/// A dense weight gradient that strides, pads and dilates at once.
///
/// Sized so that laying the input out as columns is the candidate autotune
/// picks: a small version of this test would assert against a path it never
/// takes. The options are asymmetric so that the two spatial axes disagree, and
/// an axis handled by the wrong one of the three cannot pass by symmetry.
#[test]
fn conv2d_weight_backward_dense_strided_padded_dilated_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let x = TestTensor::<4>::random([2, 5, 97, 99], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([7, 5, 3, 2], Distribution::Default, &device);
    let output_grad = TestTensor::<4>::random([2, 7, 49, 49], Distribution::Default, &device);

    let x_ref = TestTensor::<4>::from_data(x.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);
    let output_grad_ref = TestTensor::<4>::from_data(output_grad.to_data(), &ref_device);

    let options = ConvOptions::new([2, 2], [2, 1], [2, 3], 1);

    let output = module::conv2d_weight_backward(x, weight, output_grad, options.clone());
    let output_ref = module::conv2d_weight_backward(x_ref, weight_ref, output_grad_ref, options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

/// A kernel that reaches past the padded image, so some taps cover no output at
/// all.
///
/// Over the `5`-tall axis, `4` of dilation against `2` of padding leaves the
/// first and last of the three taps reading entirely outside the image while
/// the middle one reads inside — so the gradient is not simply zero, and a tap
/// that writes nothing has to differ from one that writes something. The other
/// axis is ordinary, so only one of the two is degenerate.
///
/// A short output is where laying out columns does not pay, so the winner here
/// is the convolution-by-the-gradient form, and it is `autotune-checks` —
/// which compares every candidate — that holds the column path to this shape.
#[test]
fn conv2d_weight_backward_dense_tap_outside_image_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let x = TestTensor::<4>::random([16, 4, 5, 257], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([5, 4, 3, 2], Distribution::Default, &device);
    let output_grad = TestTensor::<4>::random([16, 5, 1, 257], Distribution::Default, &device);

    let x_ref = TestTensor::<4>::from_data(x.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);
    let output_grad_ref = TestTensor::<4>::from_data(output_grad.to_data(), &ref_device);

    let options = ConvOptions::new([1, 1], [2, 1], [4, 2], 1);

    let output = module::conv2d_weight_backward(x, weight, output_grad, options.clone());
    let output_ref = module::conv2d_weight_backward(x_ref, weight_ref, output_grad_ref, options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}

/// A 1x1 convolution that strides and pads reads outside its own pixel, so it
/// is not the per-pixel matmul the pointwise path computes.
///
/// The shapes do not catch it: `in = 2 * padding + 1` under a stride of 2
/// returns an output the size of the input, which is what a pointwise
/// convolution looks like from the outside.
#[test]
fn conv2d_strided_1x1_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let input = TestTensor::<4>::random([2, 6, 3, 3], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([10, 6, 1, 1], Distribution::Default, &device);

    let input_ref = TestTensor::<4>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);

    let options = ConvOptions::new([2, 2], [1, 1], [1, 1], 1);

    let output = module::conv2d(input, weight, None, options.clone());
    let output_ref = module::conv2d(input_ref, weight_ref, None, options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}
