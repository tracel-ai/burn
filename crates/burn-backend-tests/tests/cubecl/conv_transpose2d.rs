use super::*;
use burn_tensor::{
    DType, Device, Distribution, TensorData, Tolerance, module, ops::ConvTransposeOptions,
};

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

#[test]
fn conv_transpose2d_half_precision_should_preserve_small_contributions() {
    let device = Device::default();
    for dtype in [DType::F16, DType::BF16] {
        if !device.supports_dtype(dtype) {
            continue;
        }

        let large = if dtype == DType::F16 { 2048.0 } else { 256.0 };
        // The negative-first sum also preserves the unit contribution in the direct
        // algorithm. The bias case catches rounding the overlap sum before adding bias.
        for (values, bias_value) in [(vec![-large, 1.0, large], 0.0), (vec![large, 1.0], -large)] {
            let len = values.len();
            let bias = (bias_value != 0.0)
                .then(|| TestTensor::<1>::from_data([bias_value], (&device, dtype)));
            let options = ConvTransposeOptions::new([1; 2], [0, len - 1], [0; 2], [1; 2], 1);
            let input = TestTensor::<4>::from_data(
                TensorData::new(values.clone(), [1, 1, 1, len]),
                (&device, dtype),
            );
            let weight = TestTensor::<4>::ones([1, 1, 1, len], (&device, dtype));
            let output = module::conv_transpose2d(input, weight, bias, options);
            let expected =
                TensorData::new(vec![values.iter().sum::<f32>() + bias_value], [1, 1, 1, 1])
                    .convert_dtype(dtype);
            output.into_data().assert_eq(&expected, true);
        }
    }
}
