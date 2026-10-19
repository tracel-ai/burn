use burn_backend::TensorData;
use burn_flex::{Flex, FlexDevice};
use burn_func::FuncTensor;

fn tensor(values: &[f32]) -> FuncTensor<Flex, 1> {
    FuncTensor::from_data(TensorData::from(values), &FlexDevice)
}

fn values(tensor: FuncTensor<Flex, 1>) -> Vec<f32> {
    tensor.into_data_sync().try_to_vec::<f32>().unwrap()
}

#[test]
fn arithmetic_operators() {
    let x = tensor(&[2.0, 4.0]);
    let y = tensor(&[1.0, 2.0]);

    assert_eq!(values(x.clone() + y.clone()), vec![3.0, 6.0]);
    assert_eq!(values(x.clone() - y.clone()), vec![1.0, 2.0]);
    assert_eq!(values(x.clone() * y.clone()), vec![2.0, 8.0]);
    assert_eq!(values(x.clone() / y.clone()), vec![2.0, 2.0]);
    assert_eq!(values(-x.clone()), vec![-2.0, -4.0]);
    assert_eq!(values(x.clone() * 2.0f64), vec![4.0, 8.0]);
    assert_eq!(values(x.clone() + 1.0f64), vec![3.0, 5.0]);
}

#[test]
fn unary_and_activation_methods() {
    let x = tensor(&[0.0, 1.0]);
    assert_eq!(values(x.clone().exp())[0], 1.0);
    assert!((values(x.clone().sigmoid())[0] - 0.5).abs() < 1e-6);
    assert_eq!(values(x.clone().relu()), vec![0.0, 1.0]);
    assert_eq!(values(tensor(&[-1.0, 2.0]).relu()), vec![0.0, 2.0]);
    assert_eq!(values(x.clone().abs()), vec![0.0, 1.0]);
    assert!((values(x.clone().tanh())[1] - 1.0f32.tanh()).abs() < 1e-6);
}

#[test]
fn reductions_and_shape_methods() {
    let x = tensor(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(values(x.clone().sum()), vec![10.0]);
    assert_eq!(values(x.clone().mean()), vec![2.5]);
    assert_eq!(values(x.clone().max()), vec![4.0]);
    assert_eq!(values(x.clone().min()), vec![1.0]);
    assert_eq!(values(x.clone().cumsum(0)), vec![1.0, 3.0, 6.0, 10.0]);
    assert_eq!(x.dims(), [4]);

    let matrix = FuncTensor::<Flex, 2>::from_data(
        TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]),
        &FlexDevice,
    );
    let transposed = matrix.clone().transpose();
    assert_eq!(
        transposed.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![1.0, 3.0, 2.0, 4.0]
    );
    let reshaped: FuncTensor<Flex, 1> = matrix.reshape([4]);
    assert_eq!(values(reshaped), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn matmul_and_linear() {
    let x = FuncTensor::<Flex, 2>::from_data(
        TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]),
        &FlexDevice,
    );
    let weight = FuncTensor::<Flex, 2>::from_data(
        TensorData::from([[2.0f32, 0.0], [0.0, 2.0]]),
        &FlexDevice,
    );
    let bias = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, -1.0]), &FlexDevice);

    let output = x.clone().linear(weight.clone(), Some(bias));
    assert_eq!(
        output.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![3.0, 3.0, 7.0, 7.0]
    );

    let matmul = x.matmul(weight);
    assert_eq!(
        matmul.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![2.0, 4.0, 6.0, 8.0]
    );
}

#[test]
fn slicing_and_cat() {
    let x = tensor(&[1.0, 2.0, 3.0]);
    let sliced = x.clone().slice(&[burn_std::Slice::new(0, Some(2), 1)]);
    assert_eq!(values(sliced), vec![1.0, 2.0]);

    let value = tensor(&[9.0]);
    let assigned = x
        .clone()
        .slice_assign(&[burn_std::Slice::new(0, Some(1), 1)], value);
    assert_eq!(values(assigned), vec![9.0, 2.0, 3.0]);

    let cat = FuncTensor::cat(vec![x.clone(), x], 0);
    assert_eq!(values(cat), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[cfg(feature = "bridge")]
#[test]
fn tensor_bridge_roundtrip() {
    let tensor = burn_tensor::Tensor::<1>::from_floats([1.0f32, 2.0, 3.0], &Default::default());
    let func = FuncTensor::<Flex, 1>::from_tensor(&tensor);
    assert_eq!(values(func.clone()), vec![1.0, 2.0, 3.0]);

    let roundtrip = func.into_tensor();
    assert_eq!(
        roundtrip.into_data().try_to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}
