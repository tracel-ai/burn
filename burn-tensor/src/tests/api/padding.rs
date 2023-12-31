#[burn_tensor_testgen::testgen(padding)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Int, PadMode, PadSize, Shape, Tensor};

    #[test]
    fn padding_2d_test() {
        let unpadded_floats: [[f32; 3]; 2] = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];
        let tensor = Tensor::<TestBackend, 2>::from_floats_devauto(unpadded_floats);
        let insert_num = 1.1;

        println!("Tensor from slice: {}", tensor);

        let padded_tensor = tensor.pad(PadSize::uniform(2), PadMode::Constant(insert_num));

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

        let padded_tensor = tensor.pad(PadSize::uniform(2), PadMode::Constant(insert_num));

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
        let unpadded_floats = [
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
        let tensor = Tensor::<TestBackend, 4>::from_floats_devauto(unpadded_floats);
        let insert_num = 1.1;

        println!("Tensor from slice: {}", tensor);

        let padding = [4, 3, 2, 1];
        let padded_tensor = tensor.pad(PadSize::asymmetric(padding), PadMode::Constant(insert_num));

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
        let unpadded_ints = [
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

        let tensor = Tensor::<TestBackend, 4, Int>::from_ints_devauto(unpadded_ints);
        let insert_num = 6;

        println!("Tensor from slice: {}", tensor);

        let padding = [4, 3, 2, 1];
        let padded_tensor = tensor.pad(PadSize::asymmetric(padding), PadMode::Constant(insert_num));

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
