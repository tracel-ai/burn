use super::*;
use burn_tensor::Tolerance;
use burn_tensor::module::deform_conv2d;
use burn_tensor::ops::DeformConvOptions;
use burn_tensor::{Shape, Tensor};

#[test]
fn test_deform_conv2d_simple() {
    let test = DeformConv2dTestCase {
        batch_size: 1,
        channels_in: 3,
        channels_out: 5,
        kernel_size_1: 3,
        kernel_size_2: 3,
        padding_1: 0,
        padding_2: 0,
        stride_1: 1,
        stride_2: 1,
        dilation_1: 1,
        dilation_2: 1,
        weight_groups: 1,
        offset_groups: 1,
        height: 4,
        width: 4,
    };

    test.assert_output(TestTensor::<4>::from([[
        [[0.9074, 0.6387], [0.5160, 0.4196]],
        [[2.4259, 1.8008], [1.5449, 1.3112]],
        [[3.9444, 2.9629], [2.5738, 2.2027]],
        [[5.4629, 4.1250], [3.6027, 3.0943]],
        [[6.9814, 5.2871], [4.6316, 3.9859]],
    ]]));
}

#[test]
fn test_deform_conv2d_batched() {
    let test = DeformConv2dTestCase {
        batch_size: 2,
        channels_in: 3,
        channels_out: 5,
        kernel_size_1: 3,
        kernel_size_2: 3,
        padding_1: 0,
        padding_2: 0,
        stride_1: 1,
        stride_2: 1,
        dilation_1: 1,
        dilation_2: 1,
        weight_groups: 1,
        offset_groups: 1,
        height: 4,
        width: 4,
    };

    test.assert_output(TestTensor::<4>::from([
        [
            [[0.215466, 0.192846], [0.193407, 0.175496]],
            [[0.725073, 0.675926], [0.687746, 0.648506]],
            [[1.234679, 1.159006], [1.182085, 1.121516]],
            [[1.744286, 1.642086], [1.676423, 1.594526]],
            [[2.253892, 2.125167], [2.170762, 2.067536]],
        ],
        [
            [[1.652976, 1.136937], [0.984030, 0.718403]],
            [[4.836801, 3.472453], [3.177263, 2.418021]],
            [[8.020626, 5.807969], [5.370497, 4.117639]],
            [[11.204453, 8.143486], [7.563731, 5.817256]],
            [[14.388277, 10.479003], [9.756965, 7.516875]],
        ],
    ]))
}

#[test]
fn test_deform_conv2d_weight_groups() {
    let test = DeformConv2dTestCase {
        batch_size: 1,
        channels_in: 3,
        channels_out: 6,
        kernel_size_1: 3,
        kernel_size_2: 3,
        padding_1: 0,
        padding_2: 0,
        stride_1: 1,
        stride_2: 1,
        dilation_1: 1,
        dilation_2: 1,
        weight_groups: 3,
        offset_groups: 1,
        height: 4,
        width: 4,
    };

    test.assert_output(TestTensor::<4>::from([[
        [[0.101823, 0.065756], [0.046691, 0.036233]],
        [[0.412523, 0.336674], [0.306863, 0.282386]],
        [[1.307585, 1.024152], [0.902454, 0.800008]],
        [[1.840507, 1.458072], [1.299371, 1.158781]],
        [[3.402235, 2.634555], [2.305198, 2.014265]],
        [[4.157379, 3.231476], [2.838861, 2.485659]],
    ]]))
}

#[test]
fn test_deform_conv2d_offset_groups() {
    let test = DeformConv2dTestCase {
        batch_size: 1,
        channels_in: 3,
        channels_out: 6,
        kernel_size_1: 3,
        kernel_size_2: 3,
        padding_1: 0,
        padding_2: 0,
        stride_1: 1,
        stride_2: 1,
        dilation_1: 1,
        dilation_2: 1,
        weight_groups: 1,
        offset_groups: 3,
        height: 4,
        width: 4,
    };

    test.assert_output(TestTensor::<4>::from([[
        [[1.0794, 0.7676], [0.7209, 0.5337]],
        [[2.7059, 2.0216], [1.9740, 1.5419]],
        [[4.3325, 3.2755], [3.2271, 2.5501]],
        [[5.9590, 4.5295], [4.4802, 3.5582]],
        [[7.5855, 5.7835], [5.7333, 4.5664]],
        [[9.2120, 7.0375], [6.9864, 5.5746]],
    ]]))
}

#[test]
fn test_deform_conv2d_different_kernel_size() {
    let test = DeformConv2dTestCase {
        batch_size: 1,
        channels_in: 2,
        channels_out: 3,
        kernel_size_1: 3,
        kernel_size_2: 4,
        padding_1: 0,
        padding_2: 0,
        stride_1: 1,
        stride_2: 1,
        dilation_1: 1,
        dilation_2: 1,
        weight_groups: 1,
        offset_groups: 1,
        height: 4,
        width: 4,
    };

    test.assert_output(TestTensor::<4>::from([[
        [[1.0669], [0.6329]],
        [[2.9741], [2.0383]],
        [[4.8812], [3.4437]],
    ]]))
}

#[test]
fn test_deform_conv2d_different_padding_size() {
    let test = DeformConv2dTestCase {
        batch_size: 1,
        channels_in: 2,
        channels_out: 3,
        kernel_size_1: 3,
        kernel_size_2: 3,
        padding_1: 2,
        padding_2: 3,
        stride_1: 1,
        stride_2: 1,
        dilation_1: 1,
        dilation_2: 1,
        weight_groups: 1,
        offset_groups: 1,
        height: 4,
        width: 4,
    };

    test.assert_output(TestTensor::<4>::from([[
        [
            [
                0.199779, 0.376176, 0.528501, 0.605256, 0.384365, 0.198675, 0.048145, 0.000000,
            ],
            [
                0.287923, 0.551719, 0.777562, 0.890479, 0.580469, 0.304325, 0.079554, 0.000000,
            ],
            [
                0.372947, 0.721405, 1.013668, 1.151988, 0.756444, 0.393098, 0.101582, 0.000000,
            ],
            [
                0.132138, 0.324872, 0.495372, 0.584617, 0.453122, 0.250084, 0.075703, 0.000000,
            ],
            [
                0.059332, 0.160658, 0.244789, 0.297057, 0.239464, 0.132701, 0.047114, 0.000000,
            ],
            [
                0.014338, 0.051338, 0.078303, 0.094190, 0.081278, 0.041954, 0.014506, 0.000000,
            ],
        ],
        [
            [
                0.766652, 1.164805, 1.521938, 1.711110, 1.230500, 0.807579, 0.450423, 0.333333,
            ],
            [
                0.981162, 1.601005, 2.152534, 2.440920, 1.745547, 1.091843, 0.536749, 0.333333,
            ],
            [
                1.196386, 2.044845, 2.785330, 3.152243, 2.242613, 1.351308, 0.604905, 0.333333,
            ],
            [
                0.669465, 1.178133, 1.644096, 1.902188, 1.573183, 1.033924, 0.553577, 0.333333,
            ],
            [
                0.495048, 0.786124, 1.039796, 1.204721, 1.052342, 0.743887, 0.483380, 0.333333,
            ],
            [
                0.378767, 0.498209, 0.592867, 0.654230, 0.615487, 0.488202, 0.390890, 0.333333,
            ],
        ],
        [
            [
                1.333524, 1.953435, 2.515375, 2.816964, 2.076636, 1.416483, 0.852701, 0.666667,
            ],
            [
                1.674402, 2.650291, 3.527507, 3.991360, 2.910625, 1.879361, 0.993943, 0.666667,
            ],
            [
                2.019825, 3.368286, 4.556992, 5.152499, 3.728782, 2.309520, 1.108229, 0.666667,
            ],
            [
                1.206791, 2.031395, 2.792820, 3.219759, 2.693245, 1.817763, 1.031452, 0.666667,
            ],
            [
                0.930765, 1.411590, 1.834802, 2.112385, 1.865221, 1.355072, 0.919646, 0.666667,
            ],
            [
                0.743195, 0.945081, 1.107431, 1.214270, 1.149695, 0.934451, 0.767274, 0.666667,
            ],
        ],
    ]]))
}

#[test]
fn test_deform_conv2d_different_stride() {
    let test = DeformConv2dTestCase {
        batch_size: 1,
        channels_in: 2,
        channels_out: 3,
        kernel_size_1: 3,
        kernel_size_2: 3,
        padding_1: 0,
        padding_2: 0,
        stride_1: 1,
        stride_2: 2,
        dilation_1: 1,
        dilation_2: 1,
        weight_groups: 1,
        offset_groups: 1,
        height: 4,
        width: 4,
    };

    test.assert_output(TestTensor::<4>::from([[
        [[1.0647], [0.5783]],
        [[2.9289], [1.8829]],
        [[4.7931], [3.1875]],
    ]]))
}

#[test]
fn test_deform_conv2d_different_dilation() {
    let test = DeformConv2dTestCase {
        batch_size: 1,
        channels_in: 2,
        channels_out: 3,
        kernel_size_1: 3,
        kernel_size_2: 3,
        padding_1: 0,
        padding_2: 0,
        stride_1: 1,
        stride_2: 1,
        dilation_1: 1,
        dilation_2: 2,
        weight_groups: 1,
        offset_groups: 1,
        height: 5,
        width: 5,
    };

    test.assert_output(TestTensor::<4>::from([[
        [[0.6162], [0.7611], [0.4666]],
        [[1.8578], [2.2684], [1.6208]],
        [[3.0994], [3.7757], [2.7749]],
    ]]))
}

#[test]
fn test_deform_conv2d_different_width() {
    let test = DeformConv2dTestCase {
        batch_size: 1,
        channels_in: 2,
        channels_out: 3,
        kernel_size_1: 3,
        kernel_size_2: 3,
        padding_1: 0,
        padding_2: 0,
        stride_1: 1,
        stride_2: 1,
        dilation_1: 1,
        dilation_2: 1,
        weight_groups: 1,
        offset_groups: 1,
        height: 6,
        width: 4,
    };

    test.assert_output(TestTensor::<4>::from([[
        [
            [0.8909, 0.6016],
            [1.0697, 0.7186],
            [1.2618, 0.8433],
            [0.6424, 0.5032],
        ],
        [
            [2.4670, 1.8168],
            [2.9529, 2.1497],
            [3.4805, 2.5090],
            [2.0925, 1.7411],
        ],
        [
            [4.0432, 3.0321],
            [4.8362, 3.5809],
            [5.6992, 4.1746],
            [3.5425, 2.9790],
        ],
    ]]))
}

struct DeformConv2dTestCase {
    batch_size: usize,
    channels_in: usize,
    channels_out: usize,
    kernel_size_1: usize,
    kernel_size_2: usize,
    padding_1: usize,
    padding_2: usize,
    stride_1: usize,
    stride_2: usize,
    dilation_1: usize,
    dilation_2: usize,
    weight_groups: usize,
    offset_groups: usize,
    height: usize,
    width: usize,
}

impl DeformConv2dTestCase {
    fn assert_output(self, y: Tensor<TestBackend, 4>) {
        let out_height =
            (self.height + 2 * self.padding_1 - self.dilation_1 * (self.kernel_size_1 - 1) - 1)
                / self.stride_1
                + 1;
        let out_width =
            (self.width + 2 * self.padding_2 - self.dilation_2 * (self.kernel_size_2 - 1) - 1)
                / self.stride_2
                + 1;

        let shape_x = Shape::new([self.batch_size, self.channels_in, self.height, self.width]);
        let shape_weight = Shape::new([
            self.channels_out,
            self.channels_in / self.weight_groups,
            self.kernel_size_1,
            self.kernel_size_2,
        ]);
        let shape_offset = Shape::new([
            self.batch_size,
            self.kernel_size_1 * self.kernel_size_2 * self.offset_groups * 2,
            out_height,
            out_width,
        ]);
        let shape_mask = Shape::new([
            self.batch_size,
            self.kernel_size_1 * self.kernel_size_2 * self.offset_groups,
            out_height,
            out_width,
        ]);
        let device = Default::default();
        let weight = TestTensor::<4>::from(
            TestTensorInt::arange(0..shape_weight.num_elements() as i64, &device)
                .reshape::<4, _>(shape_weight.clone())
                .into_data(),
        )
        .div_scalar(shape_weight.num_elements() as f32);
        let bias = TestTensor::<1>::from(
            TestTensorInt::arange(0..self.channels_out as i64, &device).into_data(),
        )
        .div_scalar(self.channels_out as f32);
        let x = TestTensor::<4>::from(
            TestTensorInt::arange(0..shape_x.num_elements() as i64, &device)
                .reshape::<4, _>(shape_x.clone())
                .into_data(),
        )
        .div_scalar(shape_x.num_elements() as f32);
        let offset = TestTensor::<4>::from(
            TestTensorInt::arange(0..shape_offset.num_elements() as i64, &device)
                .reshape::<4, _>(shape_offset.clone())
                .into_data(),
        )
        .div_scalar(shape_offset.num_elements() as f32);
        let mask = TestTensor::<4>::from(
            TestTensorInt::arange(0..shape_mask.num_elements() as i64, &device)
                .reshape::<4, _>(shape_mask.clone())
                .into_data(),
        )
        .div_scalar(shape_mask.num_elements() as f32);

        let output = deform_conv2d(
            x,
            offset,
            weight,
            Some(mask),
            Some(bias),
            DeformConvOptions::new(
                [self.stride_1, self.stride_2],
                [self.padding_1, self.padding_2],
                [self.dilation_1, self.dilation_2],
                self.weight_groups,
                self.offset_groups,
            ),
        );

        let tolerance = Tolerance::permissive();
        y.to_data()
            .assert_approx_eq::<FloatElem>(&output.into_data(), tolerance);
    }
}
