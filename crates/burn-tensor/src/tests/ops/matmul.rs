#[burn_tensor_testgen::testgen(matmul)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, Tensor};

    #[test]
    fn test_matmul_d2() {
        let device = Default::default();
        let tensor_1 = TestTensor::from_floats([[1.0, 7.0], [2.0, 3.0], [1.0, 5.0]], &device);
        let tensor_2 = TestTensor::from_floats([[4.0, 7.0, 5.0], [2.0, 3.0, 5.0]], &device);

        let tensor_3 = tensor_1.matmul(tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([[18.0, 28.0, 40.0], [14.0, 23.0, 25.0], [14.0, 22.0, 30.0]])
        );
    }

    #[test]
    fn test_matmul_d3() {
        let device = Default::default();
        let tensor_1 = TestTensor::from_floats([[[1.0, 7.0], [2.0, 3.0]]], &device);
        let tensor_2 = TestTensor::from_floats([[[4.0, 7.0], [2.0, 3.0]]], &device);

        let tensor_3 = tensor_1.matmul(tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([[[18.0, 28.0], [14.0, 23.0]]])
        );
    }

    #[test]
    fn test_matmul_broadcast_1() {
        let device = Default::default();
        let tensor_1 = TestTensor::from_floats([[[1.0, 7.0], [2.0, 3.0]]], &device);
        let tensor_2 = TestTensor::from_floats(
            [[[4.0, 7.0], [2.0, 3.0]], [[2.0, 5.0], [6.0, 3.0]]],
            &device,
        );

        let tensor_3 = tensor_1.matmul(tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([[[18.0, 28.0], [14.0, 23.0]], [[44.0, 26.0], [22.0, 19.0]]])
        );
    }

    #[test]
    fn test_matmul_broadcast_4d() {
        let device = Default::default();
        // [2, 1, 2, 2]
        let tensor_1 = TestTensor::from_floats(
            [[[[1.0, 7.0], [2.0, 3.0]]], [[[2.0, 5.0], [6.0, 3.0]]]],
            &device,
        );
        // [1, 2, 2, 2]
        let tensor_2 = TestTensor::from_floats(
            [[[[9.0, 8.0], [1.0, 4.0]], [[2.0, 7.0], [3.0, 5.0]]]],
            &device,
        );

        // [2, 1, 2, 2] @ [1, 2, 2, 2] -> [2, 2, 2, 2]
        let tensor_3 = tensor_1.matmul(tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([
                [[[16.0, 36.0], [21.0, 28.0]], [[23.0, 42.0], [13.0, 29.0]]],
                [[[23.0, 36.0], [57.0, 60.0]], [[19.0, 39.0], [21.0, 57.0]]]
            ])
        )
    }

    #[test]
    fn test_matmul_simple_1() {
        let device = Default::default();
        let tensor_1 = TestTensor::from_floats([[5.0, 14.0], [14.0, 50.0]], &device);
        let tensor_2 = TestTensor::from_floats([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]], &device);

        let tensor_3 = tensor_1.matmul(tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([[15.0, 34.0, 53.0], [42.0, 106.0, 170.0]])
        );
    }

    #[test]
    fn test_matmul_tmp() {
        let device = Default::default();
        let tensor_1 = TestTensor::from_floats(
            [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]],
            &device,
        );
        let tensor_2 = TestTensor::from_floats(
            [[0., 1., 2.], [4., 5., 6.], [8., 9., 10.], [12., 13., 14.]],
            &device,
        );

        let tensor_3 = tensor_1.matmul(tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([[56., 62., 68.], [152., 174., 196.], [248., 286., 324.]])
        );
    }

    #[test]
    fn test_matmul_trivial() {
        let device = Default::default();

        let tensor_1 = Tensor::<TestBackend, 1, Int>::arange(0..16, &device)
            .reshape([4, 4])
            .float();

        let tensor_3 = tensor_1.clone().matmul(tensor_1);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([
                [56., 62., 68., 74.],
                [152., 174., 196., 218.],
                [248., 286., 324., 362.],
                [344., 398., 452., 506.]
            ])
        );
    }

    #[test]
    fn test_matmul_trivial_transposed() {
        let device = Default::default();

        let tensor_1 = Tensor::<TestBackend, 1, Int>::arange(0..16, &device)
            .reshape([4, 4])
            .float();

        let tensor_3 = tensor_1.clone().matmul(tensor_1.transpose());

        assert_eq!(
            tensor_3.into_data(),
            Data::from([
                [14., 38., 62., 86.],
                [38., 126., 214., 302.],
                [62., 214., 366., 518.],
                [86., 302., 518., 734.]
            ])
        );
    }

    #[test]
    fn test_matmul_4_8() {
        let device = Default::default();

        let tensor_1 = Tensor::<TestBackend, 1, Int>::arange(0..32, &device)
            .reshape([4, 8])
            .float();

        let tensor_3 = tensor_1.clone().matmul(tensor_1.transpose());

        assert_eq!(
            tensor_3.into_data(),
            Data::from([
                [140., 364., 588., 812.],
                [364., 1100., 1836., 2572.],
                [588., 1836., 3084., 4332.],
                [812., 2572., 4332., 6092.]
            ])
        );
    }

    #[test]
    fn test_matmul_8_4() {
        let device = Default::default();

        let tensor_1 = Tensor::<TestBackend, 1, Int>::arange(0..160, &device)
            .reshape([8, 20])
            .float();
        let tensor_2 = Tensor::<TestBackend, 1, Int>::arange(0..160, &device)
            .reshape([20, 8])
            .float();

        let tensor_3 = tensor_1.clone().matmul(tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([
                [19760., 19950., 20140., 20330., 20520., 20710., 20900., 21090.],
                [50160., 50750., 51340., 51930., 52520., 53110., 53700., 54290.],
                [80560., 81550., 82540., 83530., 84520., 85510., 86500., 87490.],
                [110960., 112350., 113740., 115130., 116520., 117910., 119300., 120690.],
                [141360., 143150., 144940., 146730., 148520., 150310., 152100., 153890.],
                [171760., 173950., 176140., 178330., 180520., 182710., 184900., 187090.],
                [202160., 204750., 207340., 209930., 212520., 215110., 217700., 220290.],
                [232560., 235550., 238540., 241530., 244520., 247510., 250500., 253490.]
            ],)
        );
    }

    #[test]
    fn test_matmul_simple_2() {
        let device = Default::default();
        let tensor_1 = TestTensor::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let tensor_2 = TestTensor::from_floats([[3.0], [4.0], [5.0], [6.0]], &device);

        let tensor_3 = tensor_1.matmul(tensor_2);

        assert_eq!(tensor_3.into_data(), Data::from([[50.0]]));
    }

    #[test]
    fn test_matmul_simple_3() {
        let device = Default::default();
        let tensor_1 = TestTensor::from_floats(
            [[3., 3., 3.], [4., 4., 4.], [5., 5., 5.], [6., 6., 6.]],
            &device,
        );
        let tensor_2 = TestTensor::from_floats(
            [[1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]],
            &device,
        );

        let tensor_3 = tensor_1.matmul(tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([
                [9., 18., 27., 36.],
                [12., 24., 36., 48.],
                [15., 30., 45., 60.],
                [18., 36., 54., 72.]
            ])
        );
    }

    #[test]
    #[should_panic]
    fn should_panic_when_inner_dimensions_are_not_equal() {
        let device = Default::default();
        let tensor_1 = TestTensor::from_floats([[3., 3.], [4., 4.], [5., 5.], [6., 6.]], &device);
        let tensor_2 = TestTensor::from_floats(
            [[1., 2., 3., 4.], [1., 2., 3., 4.], [1., 2., 3., 4.]],
            &device,
        );

        let tensor_3 = tensor_1.matmul(tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([
                [9., 18., 27., 36.],
                [12., 24., 36., 48.],
                [15., 30., 45., 60.],
                [18., 36., 54., 72.]
            ])
        );
    }
}
