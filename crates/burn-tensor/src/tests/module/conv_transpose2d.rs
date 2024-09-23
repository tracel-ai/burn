#[burn_tensor_testgen::testgen(module_conv_transpose2d)]
mod tests {
    use super::*;
    use burn_tensor::module::conv_transpose2d;
    use burn_tensor::ops::ConvTransposeOptions;
    use burn_tensor::{Shape, Tensor};

    #[test]
    fn test_conv_transpose2d_simple_1() {
        let test = ConvTranspose2dTestCase {
            batch_size: 1,
            channels_in: 1,
            channels_out: 1,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 1,
            padding_out_1: 0,
            padding_out_2: 0,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            groups: 1,
            height: 2,
            width: 2,
        };

        test.assert_output(TestTensor::from([[[[5.0, 11.0], [23.0, 29.0]]]]));
    }

    #[test]
    fn test_conv_transpose2d_simple_2() {
        let test = ConvTranspose2dTestCase {
            batch_size: 1,
            channels_in: 3,
            channels_out: 3,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 1,
            padding_out_1: 0,
            padding_out_2: 0,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            groups: 1,
            height: 4,
            width: 4,
        };

        test.assert_output(TestTensor::from([[
            [
                [9855., 15207., 15738., 10797.],
                [16290., 25119., 25956., 17793.],
                [18486., 28467., 29304., 20061.],
                [13593., 20913., 21498., 14703.],
            ],
            [
                [11854., 18286., 18979., 13012.],
                [19612., 30223., 31303., 21439.],
                [22456., 34543., 35623., 24355.],
                [16456., 25288., 26035., 17782.],
            ],
            [
                [13853., 21365., 22220., 15227.],
                [22934., 35327., 36650., 25085.],
                [26426., 40619., 41942., 28649.],
                [19319., 29663., 30572., 20861.],
            ],
        ]]));
    }

    #[test]
    fn test_conv_transpose2d_simple_3() {
        let test = ConvTranspose2dTestCase {
            batch_size: 1,
            channels_in: 1,
            channels_out: 1,
            kernel_size_1: 2,
            kernel_size_2: 2,
            padding_1: 0,
            padding_2: 0,
            padding_out_1: 0,
            padding_out_2: 0,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            groups: 1,
            height: 2,
            width: 2,
        };

        test.assert_output(TestTensor::from([[[
            [0.0, 0.0, 1.0],
            [0.0, 4.0, 6.0],
            [4.0, 12.0, 9.0],
        ]]]));
    }

    #[test]
    fn test_conv_transpose2d_stride_2() {
        let test = ConvTranspose2dTestCase {
            batch_size: 1,
            channels_in: 1,
            channels_out: 1,
            kernel_size_1: 2,
            kernel_size_2: 2,
            padding_1: 0,
            padding_2: 0,
            padding_out_1: 0,
            padding_out_2: 0,
            stride_1: 2,
            stride_2: 2,
            dilation_1: 1,
            dilation_2: 1,
            groups: 1,
            height: 2,
            width: 2,
        };

        test.assert_output(TestTensor::from([[[
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0, 3.0],
            [0.0, 2.0, 0.0, 3.0],
            [4.0, 6.0, 6.0, 9.0],
        ]]]));
    }

    #[test]
    fn test_conv_transpose2d_dilation_2() {
        let test = ConvTranspose2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 2,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 1,
            padding_out_1: 1,
            padding_out_2: 1,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 2,
            dilation_2: 2,
            groups: 1,
            height: 2,
            width: 2,
        };

        test.assert_output(TestTensor::from([[
            [
                [126., 116., 136., 124., 146.],
                [108., 88., 114., 92., 120.],
                [156., 140., 166., 148., 176.],
                [126., 100., 132., 104., 138.],
                [186., 164., 196., 172., 206.],
            ],
            [
                [217., 189., 227., 197., 237.],
                [163., 125., 169., 129., 175.],
                [247., 213., 257., 221., 267.],
                [181., 137., 187., 141., 193.],
                [277., 237., 287., 245., 297.],
            ],
        ]]));
    }

    #[test]
    fn test_conv_transpose2d_stride2_out_padding() {
        let test = ConvTranspose2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 2,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 1,
            padding_out_1: 1,
            padding_out_2: 1,
            stride_1: 2,
            stride_2: 2,
            dilation_1: 1,
            dilation_2: 1,
            groups: 1,
            height: 4,
            width: 4,
        };

        test.assert_output(TestTensor::from([[
            [
                [352., 728., 378., 780., 404., 832., 430., 452.],
                [784., 1616., 836., 1720., 888., 1824., 940., 992.],
                [456., 936., 482., 988., 508., 1040., 534., 564.],
                [992., 2032., 1044., 2136., 1096., 2240., 1148., 1216.],
                [560., 1144., 586., 1196., 612., 1248., 638., 676.],
                [1200., 2448., 1252., 2552., 1304., 2656., 1356., 1440.],
                [664., 1352., 690., 1404., 716., 1456., 742., 788.],
                [784., 1598., 816., 1662., 848., 1726., 880., 926.],
            ],
            [
                [497., 1035., 541., 1123., 585., 1211., 629., 651.],
                [1145., 2373., 1233., 2549., 1321., 2725., 1409., 1461.],
                [673., 1387., 717., 1475., 761., 1563., 805., 835.],
                [1497., 3077., 1585., 3253., 1673., 3429., 1761., 1829.],
                [849., 1739., 893., 1827., 937., 1915., 981., 1019.],
                [1849., 3781., 1937., 3957., 2025., 4133., 2113., 2197.],
                [1025., 2091., 1069., 2179., 1113., 2267., 1157., 1203.],
                [1145., 2337., 1195., 2437., 1245., 2537., 1295., 1341.],
            ],
        ]]));
    }

    #[test]
    fn test_conv_transpose2d_groups_2() {
        let test = ConvTranspose2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 2,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 1,
            padding_2: 1,
            padding_out_1: 0,
            padding_out_2: 0,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            groups: 2,
            height: 2,
            width: 2,
        };

        test.assert_output(TestTensor::from([[
            [[5., 11.], [23., 29.]],
            [[236., 258.], [302., 324.]],
        ]]));
    }

    #[test]
    fn test_conv_transpose2d_groups_different_channels() {
        let test = ConvTranspose2dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 6,
            kernel_size_1: 3,
            kernel_size_2: 3,
            padding_1: 0,
            padding_2: 0,
            padding_out_1: 0,
            padding_out_2: 0,
            stride_1: 1,
            stride_2: 1,
            dilation_1: 1,
            dilation_2: 1,
            groups: 2,
            height: 2,
            width: 2,
        };

        test.assert_output(TestTensor::from([[
            [
                [0.0000e+00, 0.0000e+00, 1.0000e+00, 2.0000e+00],
                [0.0000e+00, 5.0000e+00, 1.1000e+01, 1.1000e+01],
                [6.0000e+00, 2.3000e+01, 2.9000e+01, 2.3000e+01],
                [1.2000e+01, 3.2000e+01, 3.7000e+01, 2.4000e+01],
            ],
            [
                [1.0000e+00, 1.0000e+01, 1.1000e+01, 1.2000e+01],
                [1.9000e+01, 6.0000e+01, 6.6000e+01, 4.8000e+01],
                [2.5000e+01, 7.8000e+01, 8.4000e+01, 6.0000e+01],
                [3.1000e+01, 7.8000e+01, 8.3000e+01, 5.2000e+01],
            ],
            [
                [2.0000e+00, 2.0000e+01, 2.1000e+01, 2.2000e+01],
                [3.8000e+01, 1.1500e+02, 1.2100e+02, 8.5000e+01],
                [4.4000e+01, 1.3300e+02, 1.3900e+02, 9.7000e+01],
                [5.0000e+01, 1.2400e+02, 1.2900e+02, 8.0000e+01],
            ],
            [
                [1.1100e+02, 2.5000e+02, 2.5900e+02, 1.4800e+02],
                [2.8500e+02, 6.3400e+02, 6.5600e+02, 3.6600e+02],
                [3.1500e+02, 7.0000e+02, 7.2200e+02, 4.0200e+02],
                [2.0100e+02, 4.3800e+02, 4.5100e+02, 2.4800e+02],
            ],
            [
                [1.4800e+02, 3.3200e+02, 3.4100e+02, 1.9400e+02],
                [3.7600e+02, 8.3300e+02, 8.5500e+02, 4.7500e+02],
                [4.0600e+02, 8.9900e+02, 9.2100e+02, 5.1100e+02],
                [2.5600e+02, 5.5600e+02, 5.6900e+02, 3.1200e+02],
            ],
            [
                [1.8500e+02, 4.1400e+02, 4.2300e+02, 2.4000e+02],
                [4.6700e+02, 1.0320e+03, 1.0540e+03, 5.8400e+02],
                [4.9700e+02, 1.0980e+03, 1.1200e+03, 6.2000e+02],
                [3.1100e+02, 6.7400e+02, 6.8700e+02, 3.7600e+02],
            ],
        ]]));
    }

    struct ConvTranspose2dTestCase {
        batch_size: usize,
        channels_in: usize,
        channels_out: usize,
        kernel_size_1: usize,
        kernel_size_2: usize,
        padding_1: usize,
        padding_2: usize,
        padding_out_1: usize,
        padding_out_2: usize,
        stride_1: usize,
        stride_2: usize,
        dilation_1: usize,
        dilation_2: usize,
        groups: usize,
        height: usize,
        width: usize,
    }

    impl ConvTranspose2dTestCase {
        fn assert_output(self, y: TestTensor<4>) {
            let shape_x = Shape::new([self.batch_size, self.channels_in, self.height, self.width]);
            let shape_weights = Shape::new([
                self.channels_in,
                self.channels_out / self.groups,
                self.kernel_size_1,
                self.kernel_size_2,
            ]);
            let device = Default::default();
            let weights = TestTensor::from(
                TestTensorInt::arange(0..shape_weights.num_elements() as i64, &device)
                    .reshape::<4, _>(shape_weights)
                    .into_data(),
            );
            let bias = TestTensor::from(
                TestTensorInt::arange(0..self.channels_out as i64, &device).into_data(),
            );
            let x = TestTensor::from(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &device)
                    .reshape::<4, _>(shape_x)
                    .into_data(),
            );
            let output = conv_transpose2d(
                x,
                weights,
                Some(bias),
                ConvTransposeOptions::new(
                    [self.stride_1, self.stride_2],
                    [self.padding_1, self.padding_2],
                    [self.padding_out_1, self.padding_out_2],
                    [self.dilation_1, self.dilation_2],
                    self.groups,
                ),
            );

            y.to_data().assert_approx_eq(&output.into_data(), 3);
        }
    }
}
