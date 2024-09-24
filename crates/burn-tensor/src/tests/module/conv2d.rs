#[burn_tensor_testgen::testgen(module_conv2d)]
mod tests {
    use super::*;
    use burn_tensor::module::conv2d;
    use burn_tensor::ops::ConvOptions;
    use burn_tensor::{Shape, Tensor};

    #[test]
    fn test_conv2d_simple() {
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
            width: 4,
        };

        test.assert_output(TestTensor::from([[
            [
                [1196., 1796., 1916., 1264.],
                [1881., 2793., 2946., 1923.],
                [2313., 3405., 3558., 2307.],
                [1424., 2072., 2156., 1380.],
            ],
            [
                [2709., 4173., 4509., 3065.],
                [4582., 7006., 7483., 5056.],
                [5878., 8914., 9391., 6304.],
                [4089., 6177., 6477., 4333.],
            ],
        ]]));
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

        test.assert_output(TestTensor::from([[
            [[312., 348., 384.], [492., 528., 564.], [672., 708., 744.]],
            [
                [3724., 3841., 3958.],
                [4309., 4426., 4543.],
                [4894., 5011., 5128.],
            ],
        ]]));
    }

    #[test]
    fn test_conv2d_groups_multiple_channels() {
        let test = Conv2dTestCase {
            batch_size: 1,
            channels_in: 4,
            channels_out: 4,
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

        test.assert_output(TestTensor::from([[
            [
                [4035., 4188., 4341.],
                [4800., 4953., 5106.],
                [5565., 5718., 5871.],
            ],
            [
                [10030., 10507., 10984.],
                [12415., 12892., 13369.],
                [14800., 15277., 15754.],
            ],
            [
                [56075., 56876., 57677.],
                [60080., 60881., 61682.],
                [64085., 64886., 65687.],
            ],
            [
                [78270., 79395., 80520.],
                [83895., 85020., 86145.],
                [89520., 90645., 91770.],
            ],
        ]]));
    }

    #[test]
    fn test_conv2d_complex() {
        let test = Conv2dTestCase {
            batch_size: 2,
            channels_in: 3,
            channels_out: 4,
            kernel_size_1: 3,
            kernel_size_2: 2,
            padding_1: 1,
            padding_2: 2,
            stride_1: 2,
            stride_2: 3,
            dilation_1: 1,
            dilation_2: 2,
            groups: 1,
            height: 4,
            width: 5,
        };

        test.assert_output(TestTensor::from([
            [
                [[1845., 3789., 1926.], [3210., 6465., 3228.]],
                [[4276., 9082., 4789.], [8071., 16834., 8737.]],
                [[6707., 14375., 7652.], [12932., 27203., 14246.]],
                [[9138., 19668., 10515.], [17793., 37572., 19755.]],
            ],
            [
                [[5445., 10629., 5166.], [8070., 15645., 7548.]],
                [[14356., 28882., 14509.], [22651., 45454., 22777.]],
                [[23267., 47135., 23852.], [37232., 75263., 38006.]],
                [[32178., 65388., 33195.], [51813., 105072., 53235.]],
            ],
        ]));
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

    impl Conv2dTestCase {
        fn assert_output(self, y: TestTensor<4>) {
            let shape_x = Shape::new([self.batch_size, self.channels_in, self.height, self.width]);
            let shape_weight = Shape::new([
                self.channels_out,
                self.channels_in / self.groups,
                self.kernel_size_1,
                self.kernel_size_2,
            ]);
            let device = Default::default();
            let weight = TestTensor::from(
                TestTensorInt::arange(0..shape_weight.num_elements() as i64, &device)
                    .reshape::<4, _>(shape_weight)
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
            let output = conv2d(
                x,
                weight,
                Some(bias),
                ConvOptions::new(
                    [self.stride_1, self.stride_2],
                    [self.padding_1, self.padding_2],
                    [self.dilation_1, self.dilation_2],
                    self.groups,
                ),
            );

            y.to_data().assert_approx_eq(&output.into_data(), 3);
        }
    }
}
