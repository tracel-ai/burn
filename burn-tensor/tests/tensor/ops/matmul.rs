use super::super::TestBackend;
use burn_tensor::{Data, Tensor};

#[test]
fn test_matmul_d2() {
    let data_1 = Data::from([[1.0, 7.0], [2.0, 3.0], [1.0, 5.0]]);
    let data_2 = Data::from([[4.0, 7.0, 5.0], [2.0, 3.0, 5.0]]);
    let tensor_1 = Tensor::<TestBackend, 2>::from_data(data_1.clone());
    let tensor_2 = Tensor::<TestBackend, 2>::from_data(data_2.clone());

    let tensor_3 = tensor_1.matmul(&tensor_2);

    assert_eq!(
        tensor_3.into_data(),
        Data::from([[18.0, 28.0, 40.0], [14.0, 23.0, 25.0], [14.0, 22.0, 30.0]])
    );
}

#[test]
fn test_matmul_d3() {
    let data_1 = Data::from([[[1.0, 7.0], [2.0, 3.0]]]);
    let data_2 = Data::from([[[4.0, 7.0], [2.0, 3.0]]]);
    let tensor_1 = Tensor::<TestBackend, 3>::from_data(data_1.clone());
    let tensor_2 = Tensor::<TestBackend, 3>::from_data(data_2.clone());

    let tensor_3 = tensor_1.matmul(&tensor_2);

    assert_eq!(
        tensor_3.into_data(),
        Data::from([[[18.0, 28.0], [14.0, 23.0]]])
    );
}
