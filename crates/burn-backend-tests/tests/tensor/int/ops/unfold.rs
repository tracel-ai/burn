use super::*;
use burn_tensor::Distribution;
use burn_tensor::s;
use burn_tensor::Device;

#[test]
fn test_unfold_int() {
    // Distribution::Default samples from [0, 255)
    if (IntElem::MAX as u32) < 255 - 1 {
        return;
    }
    let device = Default::default();

    let input = TestTensorInt::<3>::random([2, 6, 6], Distribution::Default, &device);

    let dim = 1;
    let size = 3;
    let step = 2;
    let actual: TestTensorInt<4> = input.clone().unfold(dim, size, step);

    let expected = TestTensorInt::<4>::empty([2, 2, 6, 3], &device)
        .slice_assign(
            s![.., 0, .., ..],
            input
                .clone()
                .slice(s![.., 0..3, ..])
                .swap_dims(1, 2)
                .unsqueeze_dim::<4>(1),
        )
        .slice_assign(
            s![.., 1, .., ..],
            input
                .clone()
                .slice(s![.., 2..5, ..])
                .swap_dims(1, 2)
                .unsqueeze_dim::<4>(1),
        );

    actual.to_data().assert_eq(&expected.to_data(), true);
}

/// The minimal failing case.
///
/// ```text
/// input            unfold(dim=1, size=2, step=2)
/// [[0,1,2,3,4],    [[[0,1],[2,3]],
///  [5,6,7,8,9]]     [[5,6],[7,8]]]
/// ```
///
/// `len = 5`, `v = 2`, so `5 % 2 = 1` and row 1 is read one element early,
/// coming back as `[[4,5],[6,7]]`.
///
/// `arange` is deliberate: every element is its own flat index, so a displaced
/// row reads as an off-by-one run rather than as arbitrary values.
///
/// # Panics
/// On an affected backend. That is the point.
pub fn minimal(device: &Device) {
    let input = TestTensorInt::arange(0..10, device).reshape([2, 5]);
    let unfolded = input.unfold::<3, _>(1, 2, 2);

    assert_eq!(unfolded.dims(), [2, 2, 2]);

    let got: Vec<i32> = unfolded.try_to_vec_as().unwrap();
    let want = vec![0, 1, 2, 3, /* row 1 */ 5, 6, 7, 8];

    assert_eq!(
        got, want,
        "\n  want {want:?}\n  got  {got:?}\n  \
         row 1 should start at flat index 5; it starts at 4, which is \
         (len / v) * v = (5 / 2) * 2.",
    );
}

/// Control: an odd `step` disables vectorization, same tail, correct result.
///
/// `size = 2`, `step = 3`, `len = 5` gives the same `num = 2` and the same
/// leftover tail of 1, but `v == 1`. Passing here is what rules out both
/// `unfold`'s stride computation and the mere presence of a tail.
///
/// # Panics
/// If this fails, the defect is *not* the one described here and the diagnosis
/// needs revisiting.
pub fn control_odd_step(device: &Device) {
    let input = TestTensorInt::arange(0..10, device).reshape([2, 5]);
    let unfolded = input.unfold::<3, _>(1, 2, 3);

    assert_eq!(unfolded.dims(), [2, 2, 2]);

    let got: Vec<i32> = unfolded.try_to_vec_as().unwrap();
    // Row 0: [0,1] [3,4]   Row 1: [5,6] [8,9]
    assert_eq!(got, vec![0, 1, 3, 4, 5, 6, 8, 9]);
}

/// Control: no leftover tail, same vectorization, correct result.
///
/// `size = 2`, `step = 2`, `len = 4` gives `tail = 0`. Passing here rules out
/// vectorization *per se*, and shows why trimming an input to the span its
/// windows cover is a sufficient workaround rather than merely a different
/// shape: `v` divides the covered span, so the truncation becomes a no-op.
///
/// # Panics
/// If this fails, the diagnosis needs revisiting.
pub fn control_no_tail(device: &Device) {
    let input = TestTensorInt::arange(0..8, device).reshape([2, 4]);
    let unfolded = input.unfold::<3, _>(1, 2, 2);

    assert_eq!(unfolded.dims(), [2, 2, 2]);

    let got: Vec<i32> = unfolded.try_to_vec_as().unwrap();
    // Row 0: [0,1] [2,3]   Row 1: [4,5] [6,7]
    assert_eq!(got, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

/// The reproduction, against the performance backend.
///
/// Ignored: it asserts the *correct* semantics, so on an affected backend
/// it fails by design. Run it to check a candidate fix.
#[test]
fn test_unfold_bug_repro() {
    let device  = Device::default();
    control_odd_step(&device);
    control_no_tail(&device);
    minimal(&device);
}

