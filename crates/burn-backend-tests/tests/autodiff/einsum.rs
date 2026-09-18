use super::*;
use burn_tensor::TensorData;

#[test]
fn einsum_backward_matrix_contraction() {
    let device = AutodiffDevice::new();
    let lhs = TestTensor::<2>::from_data([[1., 2.], [3., 4.]], &device).require_grad();
    let rhs = TestTensor::<2>::from_data([[2., 1.], [0., 3.]], &device).require_grad();

    let output: TestTensor<2> = burn_tensor::einsum!("ik,kj->ij", lhs.clone(), rhs.clone());
    let grads = output.backward();

    lhs.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([[3., 3.], [3., 3.]]), false);
    rhs.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([[4., 4.], [6., 6.]]), false);
}

#[test]
fn einsum_backward_singleton_contracted_axis() {
    let device = AutodiffDevice::new();
    let lhs = TestTensor::<2>::from_data([[2.], [3.]], &device).require_grad();
    let rhs = TestTensor::<2>::from_data([[1., 2.], [3., 4.], [5., 6.]], &device).require_grad();

    let output = TestTensor::<2>::einsum("ik,kj->ij", [lhs.clone().into(), rhs.clone().into()]);
    let grads = output.backward();

    lhs.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([[21.], [21.]]), false);
    rhs.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([[5., 5.], [5., 5.], [5., 5.]]), false);
}

#[test]
fn einsum_backward_nonadjacent_diagonal() {
    let device = AutodiffDevice::new();
    let tensor = TestTensor::<3>::from_data(
        [
            [[1., 2.], [3., 4.], [5., 6.]],
            [[7., 8.], [9., 10.], [11., 12.]],
        ],
        &device,
    )
    .require_grad();

    let output = TestTensor::<1>::einsum("iji->j", [tensor.clone().into()]);
    let grads = output.backward();

    tensor.grad(&grads).unwrap().into_data().assert_eq(
        &TensorData::from([
            [[1., 0.], [1., 0.], [1., 0.]],
            [[0., 1.], [0., 1.], [0., 1.]],
        ]),
        false,
    );
}

#[test]
fn einsum_backward_triple_diagonal() {
    let device = AutodiffDevice::new();
    let tensor = TestTensor::<3>::from_data([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], &device)
        .require_grad();

    let output = TestTensor::<1>::einsum("iii->", [tensor.clone().into()]);
    let grads = output.backward();

    tensor.grad(&grads).unwrap().into_data().assert_eq(
        &TensorData::from([[[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]]]),
        false,
    );
}

#[test]
fn einsum_backward_early_reduction() {
    let device = AutodiffDevice::new();
    let lhs = TestTensor::<2>::from_data([[1., 2.], [3., 4.]], &device).require_grad();
    let rhs = TestTensor::<1>::from_data([2., 3.], &device).require_grad();

    let output = TestTensor::<1>::einsum("ab,c->c", [lhs.clone().into(), rhs.clone().into()]);
    let grads = output.backward();

    lhs.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([[5., 5.], [5., 5.]]), false);
    rhs.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([10., 10.]), false);
}

#[test]
fn einsum_backward_delayed_reduction() {
    let device = AutodiffDevice::new();
    let a = TestTensor::<1>::from_data([1., 2.], &device).require_grad();
    let b = TestTensor::<1>::from_data([3., 4.], &device).require_grad();
    let c = TestTensor::<1>::from_data([5., 6.], &device).require_grad();

    let output = TestTensor::<1>::einsum(
        "i,i,i->",
        [a.clone().into(), b.clone().into(), c.clone().into()],
    );
    let grads = output.backward();

    a.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([15., 24.]), false);
    b.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([5., 12.]), false);
    c.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([3., 8.]), false);
}

#[test]
fn einsum_backward_broadcast_ellipsis_reduction() {
    let device = AutodiffDevice::new();
    let lhs = TestTensor::<3>::from_data([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], &device)
        .require_grad();
    let rhs = TestTensor::<2>::from_data([[2., 3.], [4., 5.]], &device).require_grad();

    let output = TestTensor::<1>::einsum("...i,...i->", [lhs.clone().into(), rhs.clone().into()]);
    let grads = output.backward();

    lhs.grad(&grads).unwrap().into_data().assert_eq(
        &TensorData::from([[[2., 3.], [4., 5.]], [[2., 3.], [4., 5.]]]),
        false,
    );
    rhs.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([[6., 8.], [10., 12.]]), false);
}

#[test]
fn einsum_backward_scalar_input() {
    let device = AutodiffDevice::new();
    let scalar = TestTensor::<1>::from_data([3.], &device).require_grad();
    let vector = TestTensor::<1>::from_data([1., 2.], &device).require_grad();

    let output = TestTensor::<1>::einsum(",i->i", [scalar.clone().into(), vector.clone().into()]);
    let grads = output.backward();

    scalar
        .grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([3.]), false);
    vector
        .grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([3., 3.]), false);
}

#[test]
fn einsum_backward_empty_contraction_keeps_both_operands_connected() {
    let device = AutodiffDevice::new();
    let lhs = TestTensor::<2>::zeros([2, 0], &device).require_grad();
    let rhs = TestTensor::<2>::zeros([0, 3], &device).require_grad();

    let output = TestTensor::<2>::einsum("ik,kj->ij", [lhs.clone().into(), rhs.clone().into()]);
    let grads = output.backward();

    lhs.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::new(Vec::<f32>::new(), [2, 0]), false);
    rhs.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::new(Vec::<f32>::new(), [0, 3]), false);
}

#[test]
fn einsum_backward_empty_output_has_zero_gradient_for_nonempty_operand() {
    let device = AutodiffDevice::new();
    let lhs = TestTensor::<2>::zeros([0, 2], &device).require_grad();
    let rhs = TestTensor::<2>::from_data([[1., 2., 3.], [4., 5., 6.]], &device).require_grad();

    let output = TestTensor::<2>::einsum("ik,kj->ij", [lhs.clone().into(), rhs.clone().into()]);
    let grads = output.backward();

    lhs.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::new(Vec::<f32>::new(), [0, 2]), false);
    rhs.grad(&grads)
        .unwrap()
        .into_data()
        .assert_eq(&TensorData::from([[0., 0., 0.], [0., 0., 0.]]), false);
}
