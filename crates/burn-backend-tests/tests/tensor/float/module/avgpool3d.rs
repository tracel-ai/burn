use super::*;
use burn_tensor::Shape;
use burn_tensor::Tolerance;
use burn_tensor::module::avg_pool3d;

#[test]
fn test_avg_pool3d_simple() {
    let test = AvgPool3dTestCase {
        batch_size: 1,
        channels: 1,
        depth: 3,
        height: 3,
        width: 3,
        kernel_size: [2, 2, 2],
        stride: [1, 1, 1],
        padding: [0, 0, 0],
        count_include_pad: true,
        ceil_mode: false,
    };

    // 3x3x3 values 0..27
    // Output shape: [1, 1, 2, 2, 2]
    // d=0:
    //   w00: (0+1+3+4+9+10+12+13)/8 = 52/8 = 6.5
    //   w01: (1+2+4+5+10+11+13+14)/8 = 60/8 = 7.5
    //   w10: (3+4+6+7+12+13+15+16)/8 = 76/8 = 9.5
    //   w11: (4+5+7+8+13+14+16+17)/8 = 84/8 = 10.5
    // d=1:
    //   w00: (9+10+12+13+18+19+21+22)/8 = 124/8 = 15.5
    //   w01: (10+11+13+14+19+20+22+23)/8 = 132/8 = 16.5
    //   w10: (12+13+15+16+21+22+24+25)/8 = 148/8 = 18.5
    //   w11: (13+14+16+17+22+23+25+26)/8 = 156/8 = 19.5
    test.assert_output(TestTensor::from([[[
        [[6.5, 7.5], [9.5, 10.5]],
        [[15.5, 16.5], [18.5, 19.5]],
    ]]]));
}

#[test]
fn test_avg_pool3d_count_include_pad() {
    let x = TestTensor::<5>::ones([1, 1, 2, 2, 2], &Default::default());

    // Pool with kernel [2, 2, 2], stride [2, 2, 2], padding [1, 1, 1]
    // With count_include_pad=true, window size is 8. Only 1 element is 1.0 (at the corner), so 1/8 = 0.125
    let output_include = avg_pool3d(x.clone(), [2, 2, 2], [2, 2, 2], [1, 1, 1], true, false);
    assert_eq!(output_include.dims(), [1, 1, 2, 2, 2]);
    let expected_include = TestTensor::<5>::from([[[
        [[0.125, 0.125], [0.125, 0.125]],
        [[0.125, 0.125], [0.125, 0.125]],
    ]]]);
    expected_include
        .to_data()
        .assert_approx_eq::<FloatElem>(&output_include.into_data(), Tolerance::default());

    // With count_include_pad=false, only unpadded elements contribute: sum / 1 = 1.0
    let output_exclude = avg_pool3d(x, [2, 2, 2], [2, 2, 2], [1, 1, 1], false, false);
    assert_eq!(output_exclude.dims(), [1, 1, 2, 2, 2]);
    let expected_exclude = TestTensor::<5>::ones([1, 1, 2, 2, 2], &Default::default());
    expected_exclude
        .to_data()
        .assert_approx_eq::<FloatElem>(&output_exclude.into_data(), Tolerance::default());
}

#[test]
fn test_avg_pool3d_ceil_mode() {
    // 5x5x5 input, kernel 2, stride 2, padding 0
    // Floor: (5 - 2)/2 + 1 = 2
    // Ceil: ceil((5 - 2)/2) + 1 = 3
    let x = TestTensor::<5>::ones([1, 1, 5, 5, 5], &Default::default());

    let out_floor = avg_pool3d(x.clone(), [2, 2, 2], [2, 2, 2], [0, 0, 0], true, false);
    assert_eq!(out_floor.dims(), [1, 1, 2, 2, 2]);

    let out_ceil = avg_pool3d(x, [2, 2, 2], [2, 2, 2], [0, 0, 0], true, true);
    assert_eq!(out_ceil.dims(), [1, 1, 3, 3, 3]);
}

#[test]
fn test_avg_pool3d_discard_branch() {
    // 5x5x5 input, kernel 2, stride 2, padding 1, ceil_mode = true
    // Window 3 would start at 2 * 3 = 6 >= 5 + 1 = 6 -> discarded!
    // Output size is 3x3x3
    let x = TestTensor::<5>::ones([1, 1, 5, 5, 5], &Default::default());
    let out = avg_pool3d(x, [2, 2, 2], [2, 2, 2], [1, 1, 1], true, true);
    assert_eq!(out.dims(), [1, 1, 3, 3, 3]);
}

struct AvgPool3dTestCase {
    batch_size: usize,
    channels: usize,
    depth: usize,
    height: usize,
    width: usize,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    count_include_pad: bool,
    ceil_mode: bool,
}

impl AvgPool3dTestCase {
    fn assert_output(self, y: TestTensor<5>) {
        let shape_x = Shape::new([
            self.batch_size,
            self.channels,
            self.depth,
            self.height,
            self.width,
        ]);
        let x = TestTensor::from(
            TestTensorInt::arange(0..shape_x.num_elements() as i64, &y.device())
                .reshape::<5, _>(shape_x)
                .into_data(),
        );
        let output = avg_pool3d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.count_include_pad,
            self.ceil_mode,
        );

        y.to_data().assert_approx_eq::<FloatElem>(
            &output.into_data(),
            Tolerance::default().set_half_precision_relative(1e-3),
        );
    }
}
