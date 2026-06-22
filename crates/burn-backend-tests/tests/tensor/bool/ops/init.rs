use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_bool_empty() {
    let shape = [2, 2];
    let tensor = TestTensorBool::<2>::empty(shape, &Default::default());
    assert_eq!(tensor.shape(), shape.into())
}

#[test]
fn should_support_bool_zeros() {
    let shape = [2, 2];
    let tensor = TestTensorBool::<2>::zeros(shape, &Default::default());
    assert_eq!(tensor.shape(), shape.into());

    tensor
        .into_data()
        .assert_eq(&TensorData::from([[false, false], [false, false]]), false);
}

#[test]
fn should_support_bool_ones() {
    let shape = [2, 2];
    let tensor = TestTensorBool::<2>::ones(shape, &Default::default());
    assert_eq!(tensor.shape(), shape.into());

    tensor
        .into_data()
        .assert_eq(&TensorData::from([[true, true], [true, true]]), false);
}

#[test]
fn should_load_bool_from_data_forcing_native_store() {
    use burn_tensor::{BoolStore, DType};

    // Model loaders (e.g. `burn-store`) reconstruct tensors with `Tensor::from_data`,
    // forcing the serialized snapshot dtype. Bool tensors are serialized as
    // `BoolStore::Native`, a storage layout that cubecl runtimes don't support. Forcing
    // it used to panic in `bool_from_data`; `resolve_dtype` must now fall back to the
    // device's default bool storage. Regression test for tracel-ai/burn#5094.
    let device = Default::default();
    let data = TensorData::new(vec![true, false, true], [3]);
    assert_eq!(data.dtype, DType::Bool(BoolStore::Native));

    let tensor = TestTensorBool::<1>::from_data(data, (&device, DType::Bool(BoolStore::Native)));

    tensor
        .into_data()
        .assert_eq(&TensorData::from([true, false, true]), false);
}
