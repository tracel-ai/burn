use burn_backend::TensorData;
use burn_flex::{Flex, FlexDevice};
use burn_func::{ForwardDiff, FuncTensor, jvp, value_and_jvp};

#[test]
fn jvp_linear_function() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32]), &device);
    let v = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);

    let f = |x: &FuncTensor<ForwardDiff<Flex>, 1>| (x.clone() * 3.0f64).sum();
    let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &v);

    assert_eq!(
        tangent.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![3.0]
    );
}

#[test]
fn value_and_jvp_broadcast() {
    let device = FlexDevice;
    let x =
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]), &device);
    let v =
        FuncTensor::<Flex, 2>::from_data(TensorData::from([[1.0f32, 0.0], [0.0, 1.0]]), &device);
    let b = FuncTensor::<Flex, 2>::from_data(TensorData::from([[2.0f32, 3.0]]), &device);
    let vb = FuncTensor::<Flex, 2>::from_data(TensorData::from([[0.5f32, -0.5]]), &device);

    let f = |(x, b): &(
        FuncTensor<ForwardDiff<Flex>, 2>,
        FuncTensor<ForwardDiff<Flex>, 2>,
    )| (x.clone() * b.clone()).sum();

    let (value, tangent): (FuncTensor<Flex, 1>, FuncTensor<Flex, 1>) =
        value_and_jvp::<Flex, _, _, _>(f, &(x, b), &(v, vb));

    assert_eq!(
        value.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![26.0]
    );
    assert_eq!(
        tangent.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![4.0]
    );
}

#[test]
fn jvp_lower_rank_broadcast() {
    let device = FlexDevice;
    let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 2.0, 3.0]), &device);
    let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 0.0, 0.0]), &device);
    let b = FuncTensor::<Flex, 2>::from_data(
        TensorData::from([[1.0f32, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        &device,
    );

    let f = |(x, b): &(
        FuncTensor<ForwardDiff<Flex>, 1>,
        FuncTensor<ForwardDiff<Flex>, 2>,
    )| { (x.clone().reshape::<2, _>([1, 3]) + b.clone()).sum() };
    // Only x has a tangent; jvp should broadcast it up to [2, 3].
    let (value, tangent): (FuncTensor<Flex, 1>, FuncTensor<Flex, 1>) = value_and_jvp::<Flex, _, _, _>(
        f,
        &(x, b),
        &(u, FuncTensor::<Flex, 2>::zeros([2, 3], &device)),
    );
    assert_eq!(
        value.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![21.0]
    );
    assert_eq!(
        tangent.into_data_sync().try_to_vec::<f32>().unwrap(),
        vec![2.0]
    );
}
