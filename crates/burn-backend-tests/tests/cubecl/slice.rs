use super::*;
use burn_tensor::Distribution;
use burn_tensor::Tolerance;

#[test]
fn slice_should_work_with_multiple_workgroups() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let tensor = TestTensor::<2>::random([6, 256], Distribution::Default, &device);
    let indices = [3..5, 45..256];
    let tensor_ref = TestTensor::<2>::from_data(tensor.to_data(), &ref_device);

    let actual = tensor.slice(indices.clone());
    let expected = tensor_ref.slice(indices);

    expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.into_data(), Tolerance::default());
}

#[test]
fn slice_prefix_with_32b_aligned_offset_end_matches_reference() {
    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    // `[1, 4, 32, 8]` with `0..7` along dim 2: `offset_start = 0`, while `offset_end` trims
    // `(32 - 7) * 8 = 200` elements (`800` bytes in f32: divisible by 32, not divisible by 64,
    // 256, or 512).
    let tensor = TestTensor::<4>::random([1, 4, 32, 8], Distribution::Default, &device);
    let tensor_ref = TestTensor::<4>::from_data(tensor.to_data(), &ref_device);
    let ranges = [0..1, 0..4, 0..7, 0..8];

    let actual = tensor.clone().slice(ranges.clone());
    let expected = tensor_ref.slice(ranges);

    // A zero-copy view shares `tensor`'s backing handle, so neither is uniquely mutable.
    #[cfg(not(feature = "fusion"))]
    assert!(
        !actual.can_mut() && !tensor.can_mut(),
        "expected 32B-aligned offset_end slice to remain a zero-copy view"
    );

    expected
        .clone()
        .into_data()
        .assert_approx_eq::<FloatElem>(&actual.clone().into_data(), Tolerance::default());

    // Downstream elementwise consumer on the 32B-aligned slice view.
    let elemwise_actual = actual.clone() * 2.0 + 1.0;
    let elemwise_expected = expected.clone() * 2.0 + 1.0;
    elemwise_expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&elemwise_actual.into_data(), Tolerance::default());

    // Downstream `Q @ K^T` matmul consumer on the 32B-aligned KV-cache prefix view.
    let q = TestTensor::<4>::random([1, 4, 1, 8], Distribution::Default, &device);
    let q_ref = TestTensor::<4>::from_data(q.to_data(), &ref_device);
    let attn_actual = q.matmul(actual.transpose());
    let attn_expected = q_ref.matmul(expected.transpose());
    attn_expected
        .into_data()
        .assert_approx_eq::<FloatElem>(&attn_actual.into_data(), Tolerance::rel_abs(1e-2, 1e-2));
}


