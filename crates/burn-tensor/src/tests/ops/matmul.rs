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

        let tensor_1 = Tensor::<TestBackend, 1, Int>::arange(0..32, &device)
            .reshape([8, 4])
            .float();
        let tensor_2 = Tensor::<TestBackend, 1, Int>::arange(0..32, &device)
            .reshape([4, 8])
            .float();

        let tensor_3 = tensor_1.clone().matmul(tensor_2);

        assert_eq!(
            tensor_3.into_data(),
            Data::from([
                [112., 118., 124., 130., 136., 142., 148., 154.],
                [304., 326., 348., 370., 392., 414., 436., 458.],
                [496., 534., 572., 610., 648., 686., 724., 762.],
                [688., 742., 796., 850., 904., 958., 1012., 1066.],
                [880., 950., 1020., 1090., 1160., 1230., 1300., 1370.],
                [1072., 1158., 1244., 1330., 1416., 1502., 1588., 1674.],
                [1264., 1366., 1468., 1570., 1672., 1774., 1876., 1978.],
                [1456., 1574., 1692., 1810., 1928., 2046., 2164., 2282.]
            ],)
        );
    //     [
    //         [112.0, 118.0, 124.0, 130.0, 880.0, 950.0, 1020.0, 1090.0],
    //         [304.0, 326.0, 348.0, 370.0, 1072.0, 1158.0, 1244.0, 1330.0],
    //         [496.0, 534.0, 572.0, 610.0, 1264.0, 1366.0, 1468.0, 1570.0],
    //         [688.0, 742.0, 796.0, 850.0, 1456.0, 1574.0, 1692.0, 1810.0],
    //         [136.0, 142.0, 148.0, 154.0, 1160.0, 1230.0, 1300.0, 1370.0],
    //         [392.0, 414.0, 436.0, 458.0, 1416.0, 1502.0, 1588.0, 1674.0],
    //         [648.0, 686.0, 724.0, 762.0, 1672.0, 1774.0, 1876.0, 1978.0],
    //         [904.0, 958.0, 1012.0, 1066.0, 1928.0, 2046.0, 2164.0, 2282.0],
    //     ]
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
