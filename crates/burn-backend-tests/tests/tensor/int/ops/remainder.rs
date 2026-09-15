use super::*;
use burn_tensor::TensorData;

#[test]
fn should_support_int_remainder_basic() {
    let data = TensorData::from([-3, -2, -1, 1, 2, 3]);
    let device = Default::default();
    let lhs = TestTensorInt::<1>::from_data(data, &device);

    let rhs = TestTensorInt::from_data(TensorData::from([2, 3, 1, 2, 1, 3]), &device);
    let output = lhs.remainder(rhs);
    let expected = TensorData::from([1, 1, -0, 1, 0, 0]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_int_remainder_broadcast() {
    let device = Default::default();
    let lhs = TestTensorInt::<2>::from_data(TensorData::from([[10, 20, 30]]), &device);
    let rhs = TestTensorInt::<2>::from_data(TensorData::from([[7]]), &device);

    let output = lhs.remainder(rhs);
    let expected = TensorData::from([[3, 6, 2]]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_int_remainder_basic_scalar() {
    let data = TensorData::from([-3, -2, -1, 1, 2, 3]);
    let device = Default::default();
    let tensor = TestTensorInt::<1>::from_data(data, &device);

    let output = tensor.remainder_scalar(2);
    let expected = TensorData::from([1, 0, 1, 1, 0, 1]);

    output.into_data().assert_eq(&expected, false);
}

#[test]
fn should_support_int_remainder_overflow() {
    // `(a % b) + b` overflows for (MAX-1) % MAX; `MIN % -1` overflows
    // before the sign fix-up.
    let device = Default::default();
    let lhs =
        TestTensorInt::<1>::from_data(TensorData::from([IntElem::MAX - 1, IntElem::MIN]), &device);
    let rhs = TestTensorInt::<1>::from_data(TensorData::from([IntElem::MAX, -1]), &device);

    lhs.remainder(rhs)
        .into_data()
        .assert_eq(&TensorData::from([IntElem::MAX - 1, 0]), false);

    TestTensorInt::<1>::from_data(TensorData::from([IntElem::MAX - 1]), &device)
        .remainder_scalar(IntElem::MAX)
        .into_data()
        .assert_eq(&TensorData::from([IntElem::MAX - 1]), false);
    TestTensorInt::<1>::from_data(TensorData::from([IntElem::MIN]), &device)
        .remainder_scalar(-1)
        .into_data()
        .assert_eq(&TensorData::from([0]), false);
}
