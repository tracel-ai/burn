#[burn_tensor_testgen::testgen(module_conv3d)]
mod tests {
    use super::*;
    use burn_tensor::module::conv3d;
    use burn_tensor::ops::ConvOptions;
    use burn_tensor::{Shape, Tensor};

    #[test]
    fn test_conv3d_simple() {
        let test = Conv3dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 2,
            kernel_size_1: 3,
            kernel_size_2: 3,
            kernel_size_3: 3,
            padding_1: 1,
            padding_2: 1,
            padding_3: 1,
            stride_1: 1,
            stride_2: 1,
            stride_3: 1,
            dilation_1: 1,
            dilation_2: 1,
            dilation_3: 1,
            groups: 1,
            depth: 4,
            height: 4,
            width: 4,
        };

        test.assert_output(TestTensor::from([[
            [
                [
                    [29980.0, 44860.0, 45640.0, 30324.0],
                    [45072.0, 67380.0, 68496.0, 45468.0],
                    [48096.0, 71844.0, 72960.0, 48396.0],
                    [31780.0, 47428.0, 48136.0, 31900.0],
                ],
                [
                    [47292.0, 70548.0, 71556.0, 47400.0],
                    [70335.0, 104823.0, 106254.0, 70317.0],
                    [74223.0, 110547.0, 111978.0, 74061.0],
                    [48552.0, 72240.0, 73140.0, 48324.0],
                ],
                [
                    [58236.0, 86676.0, 87684.0, 57960.0],
                    [85887.0, 127719.0, 129150.0, 85293.0],
                    [89775.0, 133443.0, 134874.0, 89037.0],
                    [58344.0, 86640.0, 87540.0, 57732.0],
                ],
                [
                    [36148.0, 53620.0, 54184.0, 35692.0],
                    [52740.0, 78144.0, 78936.0, 51936.0],
                    [54900.0, 81312.0, 82104.0, 54000.0],
                    [35260.0, 52156.0, 52648.0, 34580.0],
                ],
            ],
            [
                [
                    [66701.0, 100589.0, 102665.0, 68773.0],
                    [102745.0, 154861.0, 157921.0, 105733.0],
                    [110953.0, 167101.0, 170161.0, 113845.0],
                    [75413.0, 113525.0, 115529.0, 77261.0],
                ],
                [
                    [112741.0, 169693.0, 172645.0, 115441.0],
                    [172396.0, 259372.0, 263719.0, 176266.0],
                    [184060.0, 276760.0, 281107.0, 187786.0],
                    [124369.0, 186937.0, 189781.0, 126733.0],
                ],
                [
                    [144421.0, 216925.0, 219877.0, 146737.0],
                    [219052.0, 328924.0, 333271.0, 222346.0],
                    [230716.0, 346312.0, 350659.0, 233866.0],
                    [154897.0, 232441.0, 235285.0, 156877.0],
                ],
                [
                    [100517.0, 150821.0, 152681.0, 101789.0],
                    [151885.0, 227833.0, 230569.0, 153673.0],
                    [159229.0, 238777.0, 241513.0, 160921.0],
                    [106541.0, 159725.0, 161513.0, 107589.0],
                ],
            ],
        ]]));
    }

    #[test]
    fn test_conv3d_groups() {
        let test = Conv3dTestCase {
            batch_size: 1,
            channels_in: 2,
            channels_out: 2,
            kernel_size_1: 3,
            kernel_size_2: 3,
            kernel_size_3: 3,
            padding_1: 0,
            padding_2: 0,
            padding_3: 0,
            stride_1: 1,
            stride_2: 1,
            stride_3: 1,
            dilation_1: 1,
            dilation_2: 1,
            dilation_3: 1,
            groups: 2,
            depth: 5,
            height: 5,
            width: 5,
        };

        test.assert_output(TestTensor::from([[
            [
                [
                    [15219., 15570., 15921.],
                    [16974., 17325., 17676.],
                    [18729., 19080., 19431.],
                ],
                [
                    [23994., 24345., 24696.],
                    [25749., 26100., 26451.],
                    [27504., 27855., 28206.],
                ],
                [
                    [32769., 33120., 33471.],
                    [34524., 34875., 35226.],
                    [36279., 36630., 36981.],
                ],
            ],
            [
                [
                    [172819., 173899., 174979.],
                    [178219., 179299., 180379.],
                    [183619., 184699., 185779.],
                ],
                [
                    [199819., 200899., 201979.],
                    [205219., 206299., 207379.],
                    [210619., 211699., 212779.],
                ],
                [
                    [226819., 227899., 228979.],
                    [232219., 233299., 234379.],
                    [237619., 238699., 239779.],
                ],
            ],
        ]]));
    }

    #[test]
    fn test_conv3d_complex() {
        let test = Conv3dTestCase {
            batch_size: 2,
            channels_in: 3,
            channels_out: 4,
            kernel_size_1: 4,
            kernel_size_2: 3,
            kernel_size_3: 2,
            padding_1: 1,
            padding_2: 2,
            padding_3: 3,
            stride_1: 2,
            stride_2: 3,
            stride_3: 4,
            dilation_1: 1,
            dilation_2: 2,
            dilation_3: 3,
            groups: 1,
            depth: 4,
            height: 5,
            width: 6,
        };

        test.assert_output(TestTensor::from([
            [
                [
                    [[149148., 299070., 149850.], [147636., 295758., 148050.]],
                    [[150660., 301014., 150282.], [147420., 294246., 146754.]],
                ],
                [
                    [[351325., 709903., 358507.], [357589., 722143., 364483.]],
                    [[391717., 789607., 397819.], [396253., 798391., 402067.]],
                ],
                [
                    [[553502., 1120736., 567164.], [567542., 1148528., 580916.]],
                    [[632774., 1278200., 645356.], [645086., 1302536., 657380.]],
                ],
                [
                    [[755679., 1531569., 775821.], [777495., 1574913., 797349.]],
                    [[873831., 1766793., 892893.], [893919., 1806681., 912693.]],
                ],
            ],
            [
                [
                    [[408348., 810990., 402570.], [393876., 781758., 387810.]],
                    [[370980., 735174., 364122.], [354780., 702486., 347634.]],
                ],
                [
                    [
                        [1077085., 2154943., 1077787.],
                        [1070389., 2141263., 1070803.],
                    ],
                    [
                        [1078597., 2156887., 1078219.],
                        [1070173., 2139751., 1069507.],
                    ],
                ],
                [
                    [
                        [1745822., 3498896., 1753004.],
                        [1746902., 3500768., 1753796.],
                    ],
                    [
                        [1786214., 3578600., 1792316.],
                        [1785566., 3577016., 1791380.],
                    ],
                ],
                [
                    [
                        [2414559., 4842849., 2428221.],
                        [2423415., 4860273., 2436789.],
                    ],
                    [
                        [2493831., 5000313., 2506413.],
                        [2500959., 5014281., 2513253.],
                    ],
                ],
            ],
        ]));
    }

    struct Conv3dTestCase {
        batch_size: usize,
        channels_in: usize,
        channels_out: usize,
        kernel_size_1: usize,
        kernel_size_2: usize,
        kernel_size_3: usize,
        padding_1: usize,
        padding_2: usize,
        padding_3: usize,
        stride_1: usize,
        stride_2: usize,
        stride_3: usize,
        dilation_1: usize,
        dilation_2: usize,
        dilation_3: usize,
        groups: usize,
        depth: usize,
        height: usize,
        width: usize,
    }

    impl Conv3dTestCase {
        fn assert_output(self, y: TestTensor<5>) {
            let shape_x = Shape::new([
                self.batch_size,
                self.channels_in,
                self.depth,
                self.height,
                self.width,
            ]);
            let shape_weight = Shape::new([
                self.channels_out,
                self.channels_in / self.groups,
                self.kernel_size_1,
                self.kernel_size_2,
                self.kernel_size_3,
            ]);
            let device = Default::default();
            let weight = TestTensor::from(
                TestTensorInt::arange(0..shape_weight.num_elements() as i64, &device)
                    .reshape::<5, _>(shape_weight)
                    .into_data(),
            );
            let bias = TestTensor::from(
                TestTensorInt::arange(0..self.channels_out as i64, &device).into_data(),
            );
            let x = TestTensor::from(
                TestTensorInt::arange(0..shape_x.num_elements() as i64, &device)
                    .reshape::<5, _>(shape_x)
                    .into_data(),
            );
            let output = conv3d(
                x,
                weight,
                Some(bias),
                ConvOptions::new(
                    [self.stride_1, self.stride_2, self.stride_3],
                    [self.padding_1, self.padding_2, self.padding_3],
                    [self.dilation_1, self.dilation_2, self.dilation_3],
                    self.groups,
                ),
            );

            y.to_data().assert_approx_eq(&output.into_data(), 3);
        }
    }
}
