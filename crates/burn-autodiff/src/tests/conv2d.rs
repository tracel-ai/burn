#[burn_tensor_testgen::testgen(ad_conv2d)]
mod tests {
    use super::*;
    use burn_tensor::{module::conv2d, ops::ConvOptions, Data, Shape};

    #[test]
    fn test_conv2d_basic() {
        let test = Conv2dTestCase {
            batch_size: 2,
            channels_in: 2,
            channels_out: 2,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 1,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            groups: 1,
            height: 4,
            width: 4,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [
                    [
                        [
                            [88., 138., 138., 96.],
                            [150., 234., 234., 162.],
                            [150., 234., 234., 162.],
                            [112., 174., 174., 120.],
                        ],
                        [
                            [160., 246., 246., 168.],
                            [258., 396., 396., 270.],
                            [258., 396., 396., 270.],
                            [184., 282., 282., 192.],
                        ],
                    ],
                    [
                        [
                            [88., 138., 138., 96.],
                            [150., 234., 234., 162.],
                            [150., 234., 234., 162.],
                            [112., 174., 174., 120.],
                        ],
                        [
                            [160., 246., 246., 168.],
                            [258., 396., 396., 270.],
                            [258., 396., 396., 270.],
                            [184., 282., 282., 192.],
                        ],
                    ],
                ],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [
                        [[378., 516., 396.], [552., 752., 576.], [450., 612., 468.]],
                        [[666., 900., 684.], [936., 1264., 960.], [738., 996., 756.]],
                    ],
                    [
                        [[378., 516., 396.], [552., 752., 576.], [450., 612., 468.]],
                        [[666., 900., 684.], [936., 1264., 960.], [738., 996., 756.]],
                    ],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([32., 32.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv2d_different_channels() {
        let test = Conv2dTestCase {
            batch_size: 2,
            channels_in: 2,
            channels_out: 3,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 1,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            groups: 1,
            height: 4,
            width: 4,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [
                    [
                        [
                            [240., 369., 369., 252.],
                            [387., 594., 594., 405.],
                            [387., 594., 594., 405.],
                            [276., 423., 423., 288.],
                        ],
                        [
                            [348., 531., 531., 360.],
                            [549., 837., 837., 567.],
                            [549., 837., 837., 567.],
                            [384., 585., 585., 396.],
                        ],
                    ],
                    [
                        [
                            [240., 369., 369., 252.],
                            [387., 594., 594., 405.],
                            [387., 594., 594., 405.],
                            [276., 423., 423., 288.],
                        ],
                        [
                            [348., 531., 531., 360.],
                            [549., 837., 837., 567.],
                            [549., 837., 837., 567.],
                            [384., 585., 585., 396.],
                        ],
                    ],
                ],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [
                        [[378., 516., 396.], [552., 752., 576.], [450., 612., 468.]],
                        [[666., 900., 684.], [936., 1264., 960.], [738., 996., 756.]],
                    ],
                    [
                        [[378., 516., 396.], [552., 752., 576.], [450., 612., 468.]],
                        [[666., 900., 684.], [936., 1264., 960.], [738., 996., 756.]],
                    ],
                    [
                        [[378., 516., 396.], [552., 752., 576.], [450., 612., 468.]],
                        [[666., 900., 684.], [936., 1264., 960.], [738., 996., 756.]],
                    ],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([32., 32., 32.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv2d_different_kernel_size() {
        let test = Conv2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 2,
            kernel_size_1: 3,
            kernel_size_2: 4,
            padding_1: 1,
            padding_2: 1,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            groups: 1,
            height: 4,
            width: 4,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [[
                    [
                        [116., 180., 192., 132.],
                        [198., 306., 324., 222.],
                        [198., 306., 324., 222.],
                        [148., 228., 240., 164.],
                    ],
                    [
                        [212., 324., 336., 228.],
                        [342., 522., 540., 366.],
                        [342., 522., 540., 366.],
                        [244., 372., 384., 260.],
                    ],
                ]],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [
                        [
                            [27., 45., 54., 39.],
                            [52., 84., 96., 68.],
                            [51., 81., 90., 63.],
                        ],
                        [
                            [123., 189., 198., 135.],
                            [180., 276., 288., 196.],
                            [147., 225., 234., 159.],
                        ],
                    ],
                    [
                        [
                            [27., 45., 54., 39.],
                            [52., 84., 96., 68.],
                            [51., 81., 90., 63.],
                        ],
                        [
                            [123., 189., 198., 135.],
                            [180., 276., 288., 196.],
                            [147., 225., 234., 159.],
                        ],
                    ],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([12., 12.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv2d_different_padding() {
        let test = Conv2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 2,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 2,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            groups: 1,
            height: 4,
            width: 4,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [[
                    [
                        [138., 138., 138., 138.],
                        [234., 234., 234., 234.],
                        [234., 234., 234., 234.],
                        [174., 174., 174., 174.],
                    ],
                    [
                        [246., 246., 246., 246.],
                        [396., 396., 396., 396.],
                        [396., 396., 396., 396.],
                        [282., 282., 282., 282.],
                    ],
                ]],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [
                        [[66., 66., 66.], [120., 120., 120.], [114., 114., 114.]],
                        [[258., 258., 258.], [376., 376., 376.], [306., 306., 306.]],
                    ],
                    [
                        [[66., 66., 66.], [120., 120., 120.], [114., 114., 114.]],
                        [[258., 258., 258.], [376., 376., 376.], [306., 306., 306.]],
                    ],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([24., 24.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv2d_different_width() {
        let test = Conv2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 2,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 1,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            groups: 1,
            height: 4,
            width: 5,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [[
                    [
                        [88., 138., 138., 138., 96.],
                        [150., 234., 234., 234., 162.],
                        [150., 234., 234., 234., 162.],
                        [112., 174., 174., 174., 120.],
                    ],
                    [
                        [160., 246., 246., 246., 168.],
                        [258., 396., 396., 396., 270.],
                        [258., 396., 396., 396., 270.],
                        [184., 282., 282., 282., 192.],
                    ],
                ]],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [
                        [[78., 105., 90.], [144., 190., 160.], [138., 180., 150.]],
                        [[318., 405., 330.], [464., 590., 480.], [378., 480., 390.]],
                    ],
                    [
                        [[78., 105., 90.], [144., 190., 160.], [138., 180., 150.]],
                        [[318., 405., 330.], [464., 590., 480.], [378., 480., 390.]],
                    ],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([20., 20.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv2d_stride_2() {
        let test = Conv2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 2,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 1,
            stride_1: 2,
            stride_2: 2,
            dilation_1: 1,
            dilation_2: 1,
            groups: 1,
            height: 6,
            width: 6,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [[
                    [
                        [26., 52., 26., 52., 26., 28.],
                        [52., 104., 52., 104., 52., 56.],
                        [26., 52., 26., 52., 26., 28.],
                        [52., 104., 52., 104., 52., 56.],
                        [26., 52., 26., 52., 26., 28.],
                        [32., 64., 32., 64., 32., 34.],
                    ],
                    [
                        [44., 88., 44., 88., 44., 46.],
                        [88., 176., 88., 176., 88., 92.],
                        [44., 88., 44., 88., 44., 46.],
                        [88., 176., 88., 176., 88., 92.],
                        [44., 88., 44., 88., 44., 46.],
                        [50., 100., 50., 100., 50., 52.],
                    ],
                ]],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [
                        [[56., 84., 90.], [84., 126., 135.], [120., 180., 189.]],
                        [[200., 300., 306.], [300., 450., 459.], [336., 504., 513.]],
                    ],
                    [
                        [[56., 84., 90.], [84., 126., 135.], [120., 180., 189.]],
                        [[200., 300., 306.], [300., 450., 459.], [336., 504., 513.]],
                    ],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([9., 9.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv2d_different_stride() {
        let test = Conv2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 2,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 1,
            stride_1: 3,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            groups: 1,
            height: 8,
            width: 8,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [[
                    [
                        [50., 78., 78., 78., 78., 78., 78., 54.],
                        [62., 96., 96., 96., 96., 96., 96., 66.],
                        [38., 60., 60., 60., 60., 60., 60., 42.],
                        [50., 78., 78., 78., 78., 78., 78., 54.],
                        [62., 96., 96., 96., 96., 96., 96., 66.],
                        [38., 60., 60., 60., 60., 60., 60., 42.],
                        [50., 78., 78., 78., 78., 78., 78., 54.],
                        [62., 96., 96., 96., 96., 96., 96., 66.],
                    ],
                    [
                        [86., 132., 132., 132., 132., 132., 132., 90.],
                        [98., 150., 150., 150., 150., 150., 150., 102.],
                        [74., 114., 114., 114., 114., 114., 114., 78.],
                        [86., 132., 132., 132., 132., 132., 132., 90.],
                        [98., 150., 150., 150., 150., 150., 150., 102.],
                        [74., 114., 114., 114., 114., 114., 114., 78.],
                        [86., 132., 132., 132., 132., 132., 132., 90.],
                        [98., 150., 150., 150., 150., 150., 150., 102.],
                    ],
                ]],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [
                        [[434., 504., 448.], [567., 660., 588.], [735., 852., 756.]],
                        [
                            [1330., 1528., 1344.],
                            [1911., 2196., 1932.],
                            [2079., 2388., 2100.],
                        ],
                    ],
                    [
                        [[434., 504., 448.], [567., 660., 588.], [735., 852., 756.]],
                        [
                            [1330., 1528., 1344.],
                            [1911., 2196., 1932.],
                            [2079., 2388., 2100.],
                        ],
                    ],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([24., 24.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv2d_dilation_2() {
        let test = Conv2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 2,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 1,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 2,
            dilation_2: 2,
            groups: 1,
            height: 6,
            width: 6,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [[
                    [
                        [18., 38., 38., 42., 42., 22.],
                        [42., 88., 88., 96., 96., 50.],
                        [42., 88., 88., 96., 96., 50.],
                        [54., 112., 112., 120., 120., 62.],
                        [54., 112., 112., 120., 120., 62.],
                        [30., 62., 62., 66., 66., 34.],
                    ],
                    [
                        [36., 74., 74., 78., 78., 40.],
                        [78., 160., 160., 168., 168., 86.],
                        [78., 160., 160., 168., 168., 86.],
                        [90., 184., 184., 192., 192., 98.],
                        [90., 184., 184., 192., 192., 98.],
                        [48., 98., 98., 102., 102., 52.],
                    ],
                ]],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [
                        [[63., 102., 90.], [192., 280., 228.], [225., 318., 252.]],
                        [[387., 534., 414.], [624., 856., 660.], [549., 750., 576.]],
                    ],
                    [
                        [[63., 102., 90.], [192., 280., 228.], [225., 318., 252.]],
                        [[387., 534., 414.], [624., 856., 660.], [549., 750., 576.]],
                    ],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([16., 16.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv2d_different_dilation() {
        let test = Conv2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 2,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 1,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 2,
            dilation_2: 3,
            groups: 1,
            height: 6,
            width: 6,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [[
                    [
                        [18., 0., 20., 20., 0., 22.],
                        [42., 0., 46., 46., 0., 50.],
                        [42., 0., 46., 46., 0., 50.],
                        [54., 0., 58., 58., 0., 62.],
                        [54., 0., 58., 58., 0., 62.],
                        [30., 0., 32., 32., 0., 34.],
                    ],
                    [
                        [36., 0., 38., 38., 0., 40.],
                        [78., 0., 82., 82., 0., 86.],
                        [78., 0., 82., 82., 0., 86.],
                        [90., 0., 94., 94., 0., 98.],
                        [90., 0., 94., 94., 0., 98.],
                        [48., 0., 50., 50., 0., 52.],
                    ],
                ]],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [
                        [[18., 51., 33.], [60., 140., 80.], [72., 159., 87.]],
                        [[126., 267., 141.], [204., 428., 224.], [180., 375., 195.]],
                    ],
                    [
                        [[18., 51., 33.], [60., 140., 80.], [72., 159., 87.]],
                        [[126., 267., 141.], [204., 428., 224.], [180., 375., 195.]],
                    ],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([8., 8.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv2d_groups() {
        let test = Conv2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 2,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 0,
            padding_2: 0,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            groups: 2,
            height: 5,
            width: 5,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [[
                    [
                        [0., 1., 3., 3., 2.],
                        [3., 8., 15., 12., 7.],
                        [9., 21., 36., 27., 15.],
                        [9., 20., 33., 24., 13.],
                        [6., 13., 21., 15., 8.],
                    ],
                    [
                        [9., 19., 30., 21., 11.],
                        [21., 44., 69., 48., 25.],
                        [36., 75., 117., 81., 42.],
                        [27., 56., 87., 60., 31.],
                        [15., 31., 48., 33., 17.],
                    ],
                ]],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [[[54., 63., 72.], [99., 108., 117.], [144., 153., 162.]]],
                    [[[279., 288., 297.], [324., 333., 342.], [369., 378., 387.]]],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([9., 9.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv2d_groups_stride_2() {
        let test = Conv2dTestCase {
            batch_size: 1,
            channels_in: 4,
            channels_out: 4,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 1,
            stride_1: 2,
            stride_2: 2,
            dilation_1: 1,
            dilation_2: 1,
            groups: 4,
            height: 4,
            width: 4,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [[
                    [
                        [4., 8., 4., 5.],
                        [8., 16., 8., 10.],
                        [4., 8., 4., 5.],
                        [7., 14., 7., 8.],
                    ],
                    [
                        [13., 26., 13., 14.],
                        [26., 52., 26., 28.],
                        [13., 26., 13., 14.],
                        [16., 32., 16., 17.],
                    ],
                    [
                        [22., 44., 22., 23.],
                        [44., 88., 44., 46.],
                        [22., 44., 22., 23.],
                        [25., 50., 25., 26.],
                    ],
                    [
                        [31., 62., 31., 32.],
                        [62., 124., 62., 64.],
                        [31., 62., 31., 32.],
                        [34., 68., 34., 35.],
                    ],
                ]],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [[[5., 10., 12.], [10., 20., 24.], [18., 36., 40.]]],
                    [[[21., 42., 44.], [42., 84., 88.], [50., 100., 104.]]],
                    [[[37., 74., 76.], [74., 148., 152.], [82., 164., 168.]]],
                    [[[53., 106., 108.], [106., 212., 216.], [114., 228., 232.]]],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([4., 4., 4., 4.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv2d_groups_different_channels() {
        let test = Conv2dTestCase {
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
            groups: 3,
            height: 4,
            width: 4,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [[
                    [
                        [9., 20., 24., 13.],
                        [24., 52., 60., 32.],
                        [36., 76., 84., 44.],
                        [21., 44., 48., 25.],
                    ],
                    [
                        [45., 92., 96., 49.],
                        [96., 196., 204., 104.],
                        [108., 220., 228., 116.],
                        [57., 116., 120., 61.],
                    ],
                    [
                        [81., 164., 168., 85.],
                        [168., 340., 348., 176.],
                        [180., 364., 372., 188.],
                        [93., 188., 192., 97.],
                    ],
                ]],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [[[10., 14., 18.], [26., 30., 34.], [42., 46., 50.]]],
                    [[[10., 14., 18.], [26., 30., 34.], [42., 46., 50.]]],
                    [[[74., 78., 82.], [90., 94., 98.], [106., 110., 114.]]],
                    [[[74., 78., 82.], [90., 94., 98.], [106., 110., 114.]]],
                    [[[138., 142., 146.], [154., 158., 162.], [170., 174., 178.]]],
                    [[[138., 142., 146.], [154., 158., 162.], [170., 174., 178.]]],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([4., 4., 4., 4., 4., 4.], &device),
        };
        test.assert_grads(grads);
    }

    #[test]
    fn test_conv2d_complex() {
        let test = Conv2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 3,
            kernel_size_1: 2,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 2,
            stride_1: 1,
            stride_2: 2,
            dilation_1: 2,
            dilation_2: 3,
            groups: 1,
            height: 4,
            width: 5,
        };
        let device = Default::default();
        let grads = Grads {
            x: TestTensor::from_floats(
                [[
                    [
                        [36., 39., 0., 39., 42.],
                        [81., 87., 0., 87., 93.],
                        [81., 87., 0., 87., 93.],
                        [45., 48., 0., 48., 51.],
                    ],
                    [
                        [54., 57., 0., 57., 60.],
                        [117., 123., 0., 123., 129.],
                        [117., 123., 0., 123., 129.],
                        [63., 66., 0., 66., 69.],
                    ],
                ]],
                &device,
            ),
            weight: TestTensor::from_floats(
                [
                    [
                        [[15., 42., 27.], [30., 72., 42.]],
                        [[75., 162., 87.], [90., 192., 102.]],
                    ],
                    [
                        [[15., 42., 27.], [30., 72., 42.]],
                        [[75., 162., 87.], [90., 192., 102.]],
                    ],
                    [
                        [[15., 42., 27.], [30., 72., 42.]],
                        [[75., 162., 87.], [90., 192., 102.]],
                    ],
                ],
                &device,
            ),
            bias: TestTensor::from_floats([8., 8., 8.], &device),
        };
        test.assert_grads(grads);
    }

    struct Conv2dTestCase {
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
        groups: usize,
        height: usize,
        width: usize,
    }

    struct Grads {
        x: TestTensor<4>,
        weight: TestTensor<4>,
        bias: TestTensor<1>,
    }

    impl Conv2dTestCase {
        fn assert_grads(self, expected_grads: Grads) {
            let shape_x = Shape::new([self.batch_size, self.channels_in, self.height, self.width]);
            let shape_weight = Shape::new([
                self.channels_out,
                self.channels_in / self.groups,
                self.kernel_size_1,
                self.kernel_size_2,
            ]);
            let device = Default::default();
            let weight = TestAutodiffTensor::from_data(
                TestTensorInt::arange(0..shape_weight.num_elements() as i64, &device)
                    .reshape(shape_weight)
                    .into_data()
                    .convert(),
                &device,
            )
            .require_grad();
            let bias = TestAutodiffTensor::from_data(
                TestTensorInt::arange(0..self.channels_out as i64, &device)
                    .into_data()
                    .convert(),
                &device,
            )
            .require_grad();
            let x = TestAutodiffTensor::from_data(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &device)
                    .reshape(shape_x)
                    .into_data()
                    .convert(),
                &device,
            )
            .require_grad();
            let output = conv2d(
                x.clone(),
                weight.clone(),
                Some(bias.clone()),
                ConvOptions::new(
                    [self.stride_1, self.stride_2],
                    [self.padding_1, self.padding_2],
                    [self.dilation_1, self.dilation_2],
                    self.groups,
                ),
            );
            let grads = output.backward();

            // Assert
            let x_grad_actual = x.grad(&grads).unwrap();
            let weight_grad_actual = weight.grad(&grads).unwrap();
            let bias_grad_actual = bias.grad(&grads).unwrap();

            expected_grads
                .bias
                .to_data()
                .assert_approx_eq(&bias_grad_actual.to_data(), 3);
            expected_grads
                .x
                .to_data()
                .assert_approx_eq(&x_grad_actual.to_data(), 3);
            expected_grads
                .weight
                .to_data()
                .assert_approx_eq(&weight_grad_actual.to_data(), 3);
        }
    }
}
