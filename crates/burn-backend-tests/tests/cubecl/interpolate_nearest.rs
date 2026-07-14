use super::*;
use burn_tensor::ops::{InterpolateMode, InterpolateOptions};
use burn_tensor::{Distribution, Tolerance, module};

/// Regression test for https://github.com/tracel-ai/burn/issues/4686
///
/// The nearest-neighbor interpolation kernel previously used f32 division
/// to map output coordinates to input coordinates. For certain spatial
/// dimensions, GPU f32 division produced values that truncated to the
/// wrong integer, selecting the wrong input pixel.
#[test]
pub fn nearest_interpolate_should_match_reference_backend() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    // These spatial sizes previously triggered the float precision bug.
    for h in [214, 220, 227, 235, 255] {
        let tensor = TestTensor::<4>::random([1, 64, h, h], Distribution::Default, &device);
        let tensor_ref = TestTensor::<4>::from_data(tensor.to_data(), &ref_device);

        let opts = InterpolateOptions::new(InterpolateMode::Nearest);
        let out_size = [h * 2, h * 2];

        let output = module::interpolate(tensor, out_size, opts.clone());
        let output_ref = module::interpolate(tensor_ref, out_size, opts);

        output
            .into_data()
            .assert_approx_eq::<FloatElem>(&output_ref.into_data(), Tolerance::default());
    }
}
