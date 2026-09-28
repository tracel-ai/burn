use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{Device, Distribution, module};

#[test]
fn conv_transpose2d_should_match_reference_backend() {
    let device = Device::default();
    let ref_device = ReferenceDevice::new();

    device.seed(0);

    let height = 8;
    let width = 8;
    let in_channels = 8;
    let out_channels = 8;
    let batch_size = 32;
    let kernel_size_0 = 3;
    let kernel_size_1 = 3;
    let options = burn_tensor::ops::ConvTransposeOptions::new([1, 1], [1, 1], [0, 0], [1, 1], 1);

    let input = Tensor::<4>::random(
        [batch_size, in_channels, height, width],
        Distribution::Default,
        &device,
    );
    let weight = Tensor::<4>::random(
        [
            in_channels,
            out_channels / options.groups,
            kernel_size_0,
            kernel_size_1,
        ],
        Distribution::Default,
        &device,
    );
    let bias = TestTensor::<1>::random([out_channels], Distribution::Default, &device);

    let input_ref = TestTensor::<4>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);
    let bias_ref = TestTensor::<1>::from_data(bias.to_data(), &ref_device);

    let output = module::conv_transpose2d(input, weight, Some(bias), options.clone());
    let output_ref = module::conv_transpose2d(input_ref, weight_ref, Some(bias_ref), options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::rel_abs(0.01, 0.02));
}

/// A stride longer than the dilated kernel extent, so successive kernel placements leave gaps
/// and most outputs draw on a single input position.
///
/// The direct kernel used to derive its window's end from its start plus one fixed width, which
/// is empty once the stride outruns that width — every output came back zero.
#[test]
fn conv_transpose2d_stride_beyond_kernel_extent_should_match_reference_backend() {
    let device = Device::default();
    let ref_device = ReferenceDevice::new();

    device.seed(0);

    let options = burn_tensor::ops::ConvTransposeOptions::new([3, 4], [1, 0], [0, 1], [1, 2], 2);

    let input = TestTensor::<4>::random([2, 6, 5, 4], Distribution::Default, &device);
    let weight = TestTensor::<4>::random([6, 3, 2, 2], Distribution::Default, &device);
    let bias = TestTensor::<1>::random([6], Distribution::Default, &device);

    let input_ref = TestTensor::<4>::from_data(input.to_data(), &ref_device);
    let weight_ref = TestTensor::<4>::from_data(weight.to_data(), &ref_device);
    let bias_ref = TestTensor::<1>::from_data(bias.to_data(), &ref_device);

    let output = module::conv_transpose2d(input, weight, Some(bias), options.clone());
    let output_ref = module::conv_transpose2d(input_ref, weight_ref, Some(bias_ref), options);

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
}
