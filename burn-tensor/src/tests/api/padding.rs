#[burn_tensor_testgen::testgen(padding)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, PadMode, Padding, Shape, Tensor};

    #[test]
    fn padding_2d_test() {
        let unpadded_data = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];
        let data = Data::from(unpadded_data);
        let tensor = Tensor::<TestBackend, 2>::from_data_devauto(data);
        let insert_num = 1.1;

        println!("Tensor from slice: {}", tensor);

        let padding = Padding::uniform(2);
        let padded_tensor = tensor.pad(padding, PadMode::Constant, Some(insert_num));

        let padded_primitive_data_expected = [
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 0.0, 1.0, 2.0, 1.1, 1.1],
            [1.1, 1.1, 3.0, 4.0, 5.0, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
            [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
        ];

        let padded_data_expected = Data::from(padded_primitive_data_expected);
        let padded_data_actual = padded_tensor.into_data();
        assert_eq!(padded_data_expected, padded_data_actual);
    }

    #[test]
    fn padding_4d_test() {
        let unpadded_data = [
            //1
            [
                //2
                [
                    //3
                    [0.0, 1.0],
                    [2.0, 3.0],
                    [4.0, 5.0],
                ],
            ],
        ];
        let data = Data::from(unpadded_data);
        let tensor = Tensor::<TestBackend, 4>::from_data_devauto(data);
        let insert_num = 1.1;

        println!("Tensor from slice: {}", tensor);

        let padding = Padding::uniform(2);
        let padded_tensor = tensor.pad(padding, PadMode::Constant, Some(insert_num));

        let padded_primitive_data_expected = [
            //1
            [
                //2
                [
                    //3
                    [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
                    [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
                    [1.1, 1.1, 0.0, 1.0, 1.1, 1.1],
                    [1.1, 1.1, 2.0, 3.0, 1.1, 1.1],
                    [1.1, 1.1, 4.0, 5.0, 1.1, 1.1],
                    [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
                    [1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
                ],
            ],
        ];

        let padded_data_expected = Data::from(padded_primitive_data_expected);
        let padded_data_actual = padded_tensor.into_data();
        assert_eq!(padded_data_expected, padded_data_actual);
    }

    #[test]
    fn padding_asymmetric_test() {
        let unpadded_data = [
            //1
            [
                //2
                [
                    //3
                    [0.0, 1.0],
                    [2.0, 3.0],
                    [4.0, 5.0],
                ],
            ],
        ];
        let data = Data::from(unpadded_data);
        let tensor = Tensor::<TestBackend, 4>::from_data_devauto(data);
        let insert_num = 1.1;

        println!("Tensor from slice: {}", tensor);

        let padding = Padding::asymmetric([4, 3, 2, 1]);
        let padded_tensor = tensor.pad(padding, PadMode::Constant, Some(insert_num));

        let padded_primitive_data_expected = [
            //1
            [
                //2
                [
                    //3
                    [1.1, 1.1, 1.1, 1.1, 1.1],
                    [1.1, 1.1, 1.1, 1.1, 1.1],
                    [1.1, 1.1, 1.1, 1.1, 1.1],
                    [1.1, 1.1, 1.1, 1.1, 1.1],
                    [1.1, 1.1, 0.0, 1.0, 1.1],
                    [1.1, 1.1, 2.0, 3.0, 1.1],
                    [1.1, 1.1, 4.0, 5.0, 1.1],
                    [1.1, 1.1, 1.1, 1.1, 1.1],
                    [1.1, 1.1, 1.1, 1.1, 1.1],
                    [1.1, 1.1, 1.1, 1.1, 1.1],
                ],
            ],
        ];
    }

    #[test]
    fn padding_asymmetric_integer_test() {
        let unpadded_data = [
            //1
            [
                //2
                [
                    //3
                    [0, 1],
                    [2, 3],
                    [4, 5],
                ],
            ],
        ];
        let data = Data::from(unpadded_data);
        let tensor = Tensor::<TestBackend, 4, Int>::from_data_devauto(data);
        let insert_num = 6;

        println!("Tensor from slice: {}", tensor);

        let padding = Padding::asymmetric([4, 3, 2, 1]);
        let padded_tensor = tensor.pad(padding, PadMode::Constant, Some(insert_num));

        let padded_primitive_data_expected = [
            //1
            [
                //2
                [
                    //3
                    [6, 6, 6, 6, 6],
                    [6, 6, 6, 6, 6],
                    [6, 6, 6, 6, 6],
                    [6, 6, 6, 6, 6],
                    [6, 6, 0, 1, 6],
                    [6, 6, 2, 3, 6],
                    [6, 6, 4, 5, 6],
                    [6, 6, 6, 6, 6],
                    [6, 6, 6, 6, 6],
                    [6, 6, 6, 6, 6],
                ],
            ],
        ];

        let padded_data_expected = Data::from(padded_primitive_data_expected);
        let padded_data_actual = padded_tensor.into_data();
        assert_eq!(padded_data_expected, padded_data_actual);
    }

    //todo: test for additional dimensions; test for other floats
}
