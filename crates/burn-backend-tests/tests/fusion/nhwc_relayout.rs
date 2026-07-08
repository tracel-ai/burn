use super::*;
use burn_tensor::{
    Device, TensorData,
    module::{interpolate, max_pool2d},
    ops::{InterpolateMode, InterpolateOptions},
};

const BIAS: f32 = 1000.0;

fn assert_eq_data(output: TestTensor<4>, expected: impl Into<Vec<f32>>) {
    assert_eq!(output.into_data().to_vec::<f32>().unwrap(), expected.into());
}

#[test]
fn fusion_test_elementwise_operation_followed_by_max_pool2d() {
    let dev: Device = Default::default();

    let output = pool_2x2(add_zeros(seq_input(&dev), &dev));

    assert_eq_data(output, [3.0, 7.0, 11.0, 15.0]);
}

#[test]
fn fusion_test_elementwise_max_pool2d_vectorized_widths() {
    let dev: Device = Default::default();

    for dims @ [n, c, h, w] in [[2, 4, 4, 8], [1, 8, 6, 6], [3, 3, 8, 8]] {
        let flat: Vec<f32> = (0..n * c * h * w).map(|i| i as f32).collect();
        let input = TestTensor::<4>::from_data(TensorData::new(flat.clone(), dims), &dev);
        dev.sync().unwrap();

        let output = pool_2x2(input + BIAS);

        let expected = reference_pool_2x2(dims, |nn, cc, hh, ww| {
            flat[((nn * c + cc) * h + hh) * w + ww] + BIAS
        });

        assert_eq!(
            output.into_data().to_vec::<f32>().unwrap(),
            expected,
            "mismatch for shape {dims:?}"
        );
    }
}

#[test]
fn fusion_test_elementwise_max_pool2d_noncontiguous_input() {
    let dev: Device = Default::default();

    for dims @ [n, c, h, w] in [[2, 4, 4, 8], [1, 4, 6, 6]] {
        let flat: Vec<f32> = (0..n * c * h * w).map(|i| i as f32).collect();
        // Stored [N, C, W, H] then swapped to logical [N, C, H, W], making it non-contiguous.
        let base = TestTensor::<4>::from_data(TensorData::new(flat.clone(), [n, c, w, h]), &dev);
        dev.sync().unwrap();

        let output = pool_2x2(base.swap_dims(2, 3) + BIAS);

        let expected = reference_pool_2x2(dims, |nn, cc, hh, ww| {
            flat[((nn * c + cc) * w + ww) * h + hh] + BIAS
        });

        assert_eq!(
            output.into_data().to_vec::<f32>().unwrap(),
            expected,
            "mismatch for non-contiguous shape {dims:?}"
        );
    }
}

#[test]
fn fusion_test_elementwise_operation_followed_by_interpolate_nearest() {
    let dev: Device = Default::default();

    let input = add_zeros(seq_input(&dev), &dev);

    let output = interpolate(
        input,
        [4, 4],
        InterpolateOptions {
            mode: InterpolateMode::Nearest,
            align_corners: false,
        },
    );

    let expected = TestTensor::<4>::from_data(
        TensorData::from([
            [
                [
                    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                    [2.0, 2.0, 3.0, 3.0, 2.0, 2.0, 3.0, 3.0],
                ],
                [
                    [4.0, 4.0, 5.0, 5.0, 4.0, 4.0, 5.0, 5.0],
                    [6.0, 6.0, 7.0, 7.0, 6.0, 6.0, 7.0, 7.0],
                ],
            ],
            [
                [
                    [8.0, 8.0, 9.0, 9.0, 8.0, 8.0, 9.0, 9.0],
                    [10.0, 10.0, 11.0, 11.0, 10.0, 10.0, 11.0, 11.0],
                ],
                [
                    [12.0, 12.0, 13.0, 13.0, 12.0, 12.0, 13.0, 13.0],
                    [14.0, 14.0, 15.0, 15.0, 14.0, 14.0, 15.0, 15.0],
                ],
            ],
        ]),
        &dev,
    );

    assert_eq!(
        output.into_data().to_vec::<f32>().unwrap(),
        expected.into_data().to_vec::<f32>().unwrap()
    );
}

#[test]
fn fusion_test_nhwc_relayout_inplace() {
    let dev: Device = Default::default();

    let input = TestTensor::<4>::from_data(
        TensorData::from([
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
        ]),
        &dev,
    );
    dev.sync().unwrap();

    let output = pool_2x2(input * 2.0);

    assert_eq_data(output, [8.0, 16.0, 24.0, 32.0]);
}

#[test]
fn fusion_test_nhwc_relayout_broadcast_scalar() {
    let dev: Device = Default::default();

    let input = TestTensor::<4>::from_data(
        TensorData::from([
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
        ]),
        &dev,
    );
    let scalar = TestTensor::<4>::from_data(TensorData::from([[[[10.0]]]]), &dev);
    dev.sync().unwrap();

    let output = pool_2x2(input + scalar);

    assert_eq_data(output, [14.0, 18.0, 22.0, 26.0]);
}

#[test]
fn fusion_test_nhwc_relayout_broadcast_different_shapes() {
    let dev: Device = Default::default();

    let a = TestTensor::<4>::from_data(
        TensorData::from([[[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]]]),
        &dev,
    );
    let b = TestTensor::<4>::from_data(TensorData::from([[[[10.0]]], [[[20.0]]]]), &dev);
    dev.sync().unwrap();

    let output = pool_2x2(a + b);

    assert_eq_data(output, [11.0, 12.0, 21.0, 22.0]);
}

fn seq_input(dev: &Device) -> TestTensor<4> {
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    TestTensor::from_data(TensorData::new(data, [2, 2, 2, 2]), dev)
}

fn add_zeros(x: TestTensor<4>, dev: &Device) -> TestTensor<4> {
    let zeros = TestTensor::zeros(x.shape(), dev);
    dev.sync().unwrap();
    x + zeros
}

fn pool_2x2(x: TestTensor<4>) -> TestTensor<4> {
    max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1], false)
}

fn reference_pool_2x2(
    [n, c, h, w]: [usize; 4],
    val: impl Fn(usize, usize, usize, usize) -> f32,
) -> Vec<f32> {
    let (oh, ow) = (h / 2, w / 2);
    let mut out = vec![0.0f32; n * c * oh * ow];
    for nn in 0..n {
        for cc in 0..c {
            for y in 0..oh {
                for x in 0..ow {
                    let mut m = f32::MIN;
                    for kh in 0..2 {
                        for kw in 0..2 {
                            m = m.max(val(nn, cc, 2 * y + kh, 2 * x + kw));
                        }
                    }
                    out[((nn * c + cc) * oh + y) * ow + x] = m;
                }
            }
        }
    }
    out
}
