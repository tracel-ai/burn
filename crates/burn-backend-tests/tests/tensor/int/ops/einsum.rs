use super::*;
use burn_tensor::TensorData;

#[test]
fn einsum_integer_matrix_multiply() {
    let device = Default::default();
    let lhs = TestTensorInt::<2>::from_ints([[1, 2, 3], [4, 5, 6]], &device);
    let rhs = TestTensorInt::<2>::from_ints([[1, 2], [3, 4], [5, 6]], &device);

    let output: TestTensorInt<2> = burn_tensor::einsum!("ik,kj->ij", lhs, rhs);

    output
        .into_data()
        .assert_eq(&TensorData::from([[22, 28], [49, 64]]), false);
}

#[test]
fn einsum_integer_diagonal_then_contract() {
    let device = Default::default();
    let tensor = TestTensorInt::<3>::from_ints([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], &device);
    let vector = TestTensorInt::<1>::from_ints([2, 3], &device);

    TestTensorInt::<1>::einsum("iij,j->i", [tensor.into(), vector.into()])
        .into_data()
        .assert_eq(&TensorData::from([8, 38]), false);
}

#[test]
fn einsum_integer_broadcasts_contracted_axis() {
    let device = Default::default();
    let lhs = TestTensorInt::<2>::from_ints([[2], [3]], &device);
    let rhs = TestTensorInt::<2>::from_ints([[1, 2], [3, 4], [5, 6]], &device);

    TestTensorInt::<2>::einsum("ik,kj->ij", [lhs.into(), rhs.into()])
        .into_data()
        .assert_eq(&TensorData::from([[18, 24], [27, 36]]), false);
}

#[test]
fn einsum_integer_delays_reduction_until_last_operand() {
    let device = Default::default();
    let a = TestTensorInt::<1>::from_ints([1, 2], &device);
    let b = TestTensorInt::<1>::from_ints([3, 4], &device);
    let c = TestTensorInt::<1>::from_ints([5, 6], &device);

    TestTensorInt::<1>::einsum("i,i,i->", [a.into(), b.into(), c.into()])
        .into_data()
        .assert_eq(&TensorData::from([63]), false);
}

#[test]
fn einsum_integer_scalar_input() {
    let device = Default::default();
    let scalar = TestTensorInt::<1>::from_ints([3], &device);
    let vector = TestTensorInt::<1>::from_ints([1, 2], &device);

    TestTensorInt::<1>::einsum(",i->i", [scalar.into(), vector.into()])
        .into_data()
        .assert_eq(&TensorData::from([3, 6]), false);
}

#[test]
fn einsum_integer_empty_contraction() {
    let device = Default::default();
    let lhs = TestTensorInt::<2>::zeros([2, 0], &device);
    let rhs = TestTensorInt::<2>::zeros([0, 3], &device);

    TestTensorInt::<2>::einsum("ik,kj->ij", [lhs.into(), rhs.into()])
        .into_data()
        .assert_eq(&TensorData::from([[0, 0, 0], [0, 0, 0]]), false);
}
