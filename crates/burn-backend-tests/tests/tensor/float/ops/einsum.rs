use super::*;
use burn_tensor::TensorData;

#[test]
fn einsum_matrix_multiply_runtime() {
    let device = Default::default();
    let lhs = TestTensor::<2>::from_data([[1., 2., 3.], [4., 5., 6.]], &device);
    let rhs = TestTensor::<2>::from_data([[1., 2.], [3., 4.], [5., 6.]], &device);
    let equation = ["ik", ",kj", "->ij"].concat();

    TestTensor::<2>::einsum(&equation, [lhs.into(), rhs.into()])
        .into_data()
        .assert_eq(&TensorData::from([[22., 28.], [49., 64.]]), false);
}

#[test]
fn einsum_macro_accepts_different_operand_ranks() {
    let device = Default::default();
    let matrix = TestTensor::<2>::from_data([[1., 2., 3.], [4., 5., 6.]], &device);
    let vector = TestTensor::<1>::from_data([2., 3., 4.], &device);

    let output: TestTensor<1> = burn_tensor::einsum!("ij,j->i", matrix, vector);

    output
        .into_data()
        .assert_eq(&TensorData::from([20., 47.]), false);
}

#[test]
fn einsum_macro_scalar_output() {
    let device = Default::default();
    let lhs = TestTensor::<1>::from_data([1., 2., 3.], &device);
    let rhs = TestTensor::<1>::from_data([4., 5., 6.], &device);

    let output: TestTensor<1> = burn_tensor::einsum!("i,i->", lhs, rhs);

    output
        .into_data()
        .assert_eq(&TensorData::from([32.]), false);
}

#[test]
fn einsum_outer_product_output_order() {
    let device = Default::default();
    let lhs = TestTensor::<1>::from_data([1., 2.], &device);
    let rhs = TestTensor::<1>::from_data([3., 4., 5.], &device);

    TestTensor::<2>::einsum("i,j->ji", [lhs.into(), rhs.into()])
        .into_data()
        .assert_eq(&TensorData::from([[3., 6.], [4., 8.], [5., 10.]]), false);
}

#[test]
fn einsum_single_operand_reductions() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1., 2., 3.], [4., 5., 6.]], &device);

    TestTensor::<1>::einsum("ij->i", [tensor.clone().into()])
        .into_data()
        .assert_eq(&TensorData::from([6., 15.]), false);
    TestTensor::<1>::einsum("ij->j", [tensor.clone().into()])
        .into_data()
        .assert_eq(&TensorData::from([5., 7., 9.]), false);
    TestTensor::<1>::einsum("ij->", [tensor.into()])
        .into_data()
        .assert_eq(&TensorData::from([21.]), false);
}

#[test]
fn einsum_reduces_independent_axes_early() {
    let device = Default::default();
    let lhs = TestTensor::<2>::from_data([[1., 2.], [3., 4.]], &device);
    let rhs = TestTensor::<1>::from_data([2., 3.], &device);

    TestTensor::<1>::einsum("ab,c->c", [lhs.into(), rhs.into()])
        .into_data()
        .assert_eq(&TensorData::from([20., 30.]), false);
}

#[test]
fn einsum_reduces_left_axis_before_contraction() {
    let device = Default::default();
    let lhs = TestTensor::<2>::from_data([[1., 2.], [3., 4.]], &device);
    let rhs = TestTensor::<2>::from_data([[2., 3.], [4., 5.]], &device);

    TestTensor::<1>::einsum("ab,bc->c", [lhs.into(), rhs.into()])
        .into_data()
        .assert_eq(&TensorData::from([32., 42.]), false);
}

#[test]
fn einsum_delays_reduction_until_last_operand() {
    let device = Default::default();
    let a = TestTensor::<1>::from_data([1., 2.], &device);
    let b = TestTensor::<1>::from_data([3., 4.], &device);
    let c = TestTensor::<1>::from_data([5., 6.], &device);

    TestTensor::<1>::einsum("i,i,i->", [a.into(), b.into(), c.into()])
        .into_data()
        .assert_eq(&TensorData::from([63.]), false);
}

#[test]
fn einsum_three_way_contraction() {
    let device = Default::default();
    let a = TestTensor::<3>::from_data([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], &device);
    let b = TestTensor::<3>::from_data([[[1., 2.], [3., 4.]], [[2., 1.], [4., 3.]]], &device);
    let c = TestTensor::<3>::ones([2, 2, 2], &device);

    TestTensor::<1>::einsum("abc,acd,bcd->", [a.into(), b.into(), c.into()])
        .into_data()
        .assert_eq(&TensorData::from([188.]), false);
}

#[test]
fn einsum_four_operand_chain() {
    let device = Default::default();
    let a = TestTensor::<2>::from_data([[1., 2.], [3., 4.]], &device);
    let b = TestTensor::<2>::from_data([[1., 0.], [0., 2.]], &device);
    let c = TestTensor::<2>::from_data([[2., 0.], [0., 1.]], &device);
    let d = TestTensor::<2>::from_data([[1., 2.], [3., 4.]], &device);

    let output: TestTensor<2> = burn_tensor::einsum!("ab,bc,cd,de->ae", a, b, c, d);

    output
        .into_data()
        .assert_eq(&TensorData::from([[14., 20.], [30., 44.]]), false);
}

#[test]
fn einsum_broadcasts_output_axes() {
    let device = Default::default();
    let lhs = TestTensor::<2>::from_data([[2.], [3.]], &device);
    let rhs = TestTensor::<2>::from_data([[4., 5., 6.]], &device);

    TestTensor::<2>::einsum("ij,ij->ij", [lhs.into(), rhs.into()])
        .into_data()
        .assert_eq(&TensorData::from([[8., 10., 12.], [12., 15., 18.]]), false);
}

#[test]
fn einsum_broadcasts_contracted_axis_on_either_operand() {
    let device = Default::default();
    let lhs = TestTensor::<2>::from_data([[2.], [3.]], &device);
    let rhs = TestTensor::<2>::from_data([[1., 2.], [3., 4.], [5., 6.]], &device);

    TestTensor::<2>::einsum("ik,kj->ij", [lhs.clone().into(), rhs.clone().into()])
        .into_data()
        .assert_eq(&TensorData::from([[18., 24.], [27., 36.]]), false);
    TestTensor::<2>::einsum("kj,ik->ij", [rhs.into(), lhs.into()])
        .into_data()
        .assert_eq(&TensorData::from([[18., 24.], [27., 36.]]), false);
}

#[test]
fn einsum_ellipsis_right_aligns_different_ranks() {
    let device = Default::default();
    let lhs = TestTensor::<4>::from_data([[[[1., 2.], [3., 4.]]], [[[5., 6.], [7., 8.]]]], &device);
    let rhs = TestTensor::<3>::from_data(
        [
            [[1., 2.], [3., 4.]],
            [[2., 1.], [4., 3.]],
            [[1., 0.], [0., 1.]],
        ],
        &device,
    );

    let output: TestTensor<4> = burn_tensor::einsum!("...ij,...jk->...ik", lhs, rhs);

    output.into_data().assert_eq(
        &TensorData::from([
            [
                [[7., 10.], [15., 22.]],
                [[10., 7.], [22., 15.]],
                [[1., 2.], [3., 4.]],
            ],
            [
                [[23., 34.], [31., 46.]],
                [[34., 23.], [46., 31.]],
                [[5., 6.], [7., 8.]],
            ],
        ]),
        false,
    );
}

#[test]
fn einsum_reduces_broadcast_ellipsis() {
    let device = Default::default();
    let lhs = TestTensor::<3>::from_data([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], &device);
    let rhs = TestTensor::<2>::from_data([[2., 3.], [4., 5.]], &device);

    TestTensor::<1>::einsum("...i,...i->", [lhs.clone().into(), rhs.into()])
        .into_data()
        .assert_eq(&TensorData::from([136.]), false);
    TestTensor::<1>::einsum("...i->i", [lhs.into()])
        .into_data()
        .assert_eq(&TensorData::from([16., 20.]), false);
}

#[test]
fn einsum_middle_and_empty_ellipsis() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], &device);
    TestTensor::<3>::einsum("i...j->...ji", [tensor.into()])
        .into_data()
        .assert_eq(
            &TensorData::from([[[1., 5.], [2., 6.]], [[3., 7.], [4., 8.]]]),
            false,
        );

    let vector = TestTensor::<1>::from_data([1., 2.], &device);
    TestTensor::<1>::einsum("...i->...i", [vector.into()])
        .into_data()
        .assert_eq(&TensorData::from([1., 2.]), false);

    let matrix = TestTensor::<2>::from_data([[1., 2.], [3., 4.]], &device);
    TestTensor::<2>::einsum("ij->...ji", [matrix.into()])
        .into_data()
        .assert_eq(&TensorData::from([[1., 3.], [2., 4.]]), false);
}

#[test]
fn einsum_implicit_output_sorts_uppercase_before_lowercase() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1., 2., 3.], [4., 5., 6.]], &device);

    let output: TestTensor<2> = burn_tensor::einsum!("bA", tensor);

    output
        .into_data()
        .assert_eq(&TensorData::from([[1., 4.], [2., 5.], [3., 6.]]), false);
}

#[test]
fn einsum_implicit_output_places_ellipsis_first() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data([[[1., 2., 3.], [4., 5., 6.]]], &device);

    TestTensor::<3>::einsum("...ji", [tensor.into()])
        .into_data()
        .assert_eq(&TensorData::from([[[1., 4.], [2., 5.], [3., 6.]]]), false);
}

#[test]
fn einsum_diagonal_and_trace() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1., 2.], [3., 4.]], &device);

    TestTensor::<1>::einsum("ii->i", [tensor.clone().into()])
        .into_data()
        .assert_eq(&TensorData::from([1., 4.]), false);
    TestTensor::<1>::einsum("ii", [tensor.into()])
        .into_data()
        .assert_eq(&TensorData::from([5.]), false);
}

#[test]
fn einsum_nonadjacent_and_triple_diagonal() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data(
        [
            [[1., 2.], [3., 4.], [5., 6.]],
            [[7., 8.], [9., 10.], [11., 12.]],
        ],
        &device,
    );
    TestTensor::<1>::einsum("iji->j", [tensor.into()])
        .into_data()
        .assert_eq(&TensorData::from([9., 13., 17.]), false);

    let tensor = TestTensor::<3>::from_data([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], &device);
    TestTensor::<1>::einsum("iii->i", [tensor.into()])
        .into_data()
        .assert_eq(&TensorData::from([1., 8.]), false);
}

#[test]
fn einsum_diagonal_then_contract() {
    let device = Default::default();
    let tensor = TestTensor::<3>::from_data([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], &device);
    let vector = TestTensor::<1>::from_data([2., 3.], &device);

    TestTensor::<1>::einsum("iij,j->i", [tensor.into(), vector.into()])
        .into_data()
        .assert_eq(&TensorData::from([8., 38.]), false);
}

#[test]
fn einsum_scalar_inputs_use_one_element_tensors() {
    let device = Default::default();
    let scalar = TestTensor::<1>::from_data([3.], &device);
    let vector = TestTensor::<1>::from_data([1., 2.], &device);

    TestTensor::<1>::einsum(",i->i", [scalar.clone().into(), vector.into()])
        .into_data()
        .assert_eq(&TensorData::from([3., 6.]), false);
    TestTensor::<1>::einsum("->", [scalar.clone().into()])
        .into_data()
        .assert_eq(&TensorData::from([3.]), false);
    TestTensor::<1>::einsum("   ", [scalar.into()])
        .into_data()
        .assert_eq(&TensorData::from([3.]), false);
}

#[test]
fn einsum_ignores_spaces_between_tokens() {
    let device = Default::default();
    let lhs = TestTensor::<2>::from_data([[1., 2.]], &device);
    let rhs = TestTensor::<2>::from_data([[3.], [4.]], &device);

    TestTensor::<2>::einsum(" i k , k j -> i j ", [lhs.into(), rhs.into()])
        .into_data()
        .assert_eq(&TensorData::from([[11.]]), false);
}

#[test]
fn einsum_noncontiguous_operands() {
    let device = Default::default();
    let lhs = TestTensor::<2>::from_data([[1., 2., 3.], [4., 5., 6.]], &device).transpose();
    let rhs = TestTensor::<2>::from_data([[1., 2.], [3., 4.], [5., 6.]], &device).transpose();

    TestTensor::<2>::einsum("ik,kj->ij", [lhs.into(), rhs.into()])
        .into_data()
        .assert_eq(
            &TensorData::from([[9., 19., 29.], [12., 26., 40.], [15., 33., 51.]]),
            false,
        );
}

#[test]
fn einsum_empty_contraction_is_zero() {
    let device = Default::default();
    let lhs = TestTensor::<2>::zeros([2, 0], &device);
    let rhs = TestTensor::<2>::zeros([0, 3], &device);

    TestTensor::<2>::einsum("ik,kj->ij", [lhs.into(), rhs.into()])
        .into_data()
        .assert_eq(&TensorData::from([[0., 0., 0.], [0., 0., 0.]]), false);
}

#[test]
fn einsum_empty_output() {
    let device = Default::default();
    let lhs = TestTensor::<2>::zeros([0, 3], &device);
    let rhs = TestTensor::<1>::from_data([1., 2., 3.], &device);

    TestTensor::<1>::einsum("ij,j->i", [lhs.into(), rhs.into()])
        .into_data()
        .assert_eq(&TensorData::new(Vec::<f32>::new(), [0]), false);
}

#[test]
fn einsum_empty_diagonal_and_trace() {
    let device = Default::default();
    let tensor = TestTensor::<2>::zeros([0, 0], &device);

    TestTensor::<1>::einsum("ii->i", [tensor.clone().into()])
        .into_data()
        .assert_eq(&TensorData::new(Vec::<f32>::new(), [0]), false);
    TestTensor::<1>::einsum("ii->", [tensor.into()])
        .into_data()
        .assert_eq(&TensorData::from([0.]), false);
}

#[test]
fn einsum_empty_ellipsis() {
    let device = Default::default();
    let tensor = TestTensor::<3>::zeros([2, 0, 4], &device);

    TestTensor::<2>::einsum("...i->...", [tensor.into()])
        .into_data()
        .assert_eq(&TensorData::new(Vec::<f32>::new(), [2, 0]), false);
}

#[test]
#[should_panic]
fn einsum_rejects_invalid_equation() {
    let device = Default::default();
    let tensor = TestTensor::<2>::ones([2, 2], &device);
    TestTensor::<1>::einsum("ij->i->j", [tensor.into()]);
}

#[test]
#[should_panic]
fn einsum_rejects_output_rank_mismatch() {
    let device = Default::default();
    let tensor = TestTensor::<1>::ones([2], &device);
    TestTensor::<2>::einsum("i->i", [tensor.into()]);
}

#[test]
#[should_panic]
fn einsum_rejects_diagonal_size_mismatch() {
    let device = Default::default();
    let tensor = TestTensor::<2>::ones([2, 3], &device);
    TestTensor::<1>::einsum("ii->i", [tensor.into()]);
}

#[test]
#[should_panic]
fn einsum_rejects_shared_label_size_mismatch() {
    let device = Default::default();
    let lhs = TestTensor::<2>::ones([2, 2], &device);
    let rhs = TestTensor::<2>::ones([3, 2], &device);
    TestTensor::<2>::einsum("ik,kj->ij", [lhs.into(), rhs.into()]);
}

#[test]
#[should_panic]
fn einsum_rejects_ellipsis_size_mismatch() {
    let device = Default::default();
    let lhs = TestTensor::<2>::ones([2, 1], &device);
    let rhs = TestTensor::<2>::ones([3, 1], &device);
    TestTensor::<1>::einsum("...i,...i->...", [lhs.into(), rhs.into()]);
}

#[test]
#[should_panic]
fn einsum_rejects_nonscalar_for_empty_subscript() {
    let device = Default::default();
    let tensor = TestTensor::<1>::ones([2], &device);
    TestTensor::<1>::einsum("->", [tensor.into()]);
}

#[test]
fn einsum_contraction_places_right_output_before_left() {
    let device = Default::default();
    let lhs = TestTensor::<2>::from_data([[1., 2., 3.], [4., 5., 6.]], &device);
    let rhs = TestTensor::<2>::from_data(
        [[1., 2., 3., 4.], [2., 3., 4., 5.], [3., 4., 5., 6.]],
        &device,
    );

    TestTensor::<2>::einsum("ik,kj->ji", [lhs.into(), rhs.into()])
        .into_data()
        .assert_eq(
            &TensorData::from([[14., 32.], [20., 47.], [26., 62.], [32., 77.]]),
            false,
        );
}

#[test]
fn einsum_matrix_multiply_shared_batch() {
    let device = Default::default();
    let lhs = TestTensor::<3>::from_data([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], &device);
    let rhs = TestTensor::<3>::from_data([[[1., 2.], [3., 4.]], [[2., 1.], [4., 3.]]], &device);

    TestTensor::<3>::einsum("bij,bjk->bik", [lhs.into(), rhs.into()])
        .into_data()
        .assert_eq(
            &TensorData::from([[[7., 10.], [15., 22.]], [[34., 23.], [46., 31.]]]),
            false,
        );
}

#[test]
fn einsum_diagonal_ignores_off_diagonal_nan() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from_data([[1., f32::NAN], [f32::NAN, 4.]], &device);

    TestTensor::<1>::einsum("ii->i", [tensor.clone().into()])
        .into_data()
        .assert_eq(&TensorData::from([1., 4.]), false);
    TestTensor::<1>::einsum("ii->", [tensor.into()])
        .into_data()
        .assert_eq(&TensorData::from([5.]), false);
}

#[test]
fn einsum_delays_broadcast_reduction_until_last_operand() {
    let device = Default::default();
    let a = TestTensor::<1>::from_data([1., 2.], &device);
    let b = TestTensor::<1>::from_data([3.], &device);
    let c = TestTensor::<1>::from_data([4., 5.], &device);

    TestTensor::<1>::einsum("i,i,i->", [a.into(), b.into(), c.into()])
        .into_data()
        .assert_eq(&TensorData::from([42.]), false);
}

#[test]
fn einsum_empty_delayed_broadcast_reduction() {
    let device = Default::default();
    let a = TestTensor::<1>::zeros([0], &device);
    let b = TestTensor::<1>::from_data([3.], &device);
    let c = TestTensor::<1>::zeros([0], &device);

    TestTensor::<1>::einsum("i,i,i->", [a.into(), b.into(), c.into()])
        .into_data()
        .assert_eq(&TensorData::from([0.]), false);
}

#[test]
#[should_panic]
fn einsum_rejects_missing_operand() {
    let device = Default::default();
    let tensor = TestTensor::<1>::ones([2], &device);
    TestTensor::<1>::einsum("i,i->i", [tensor.into()]);
}

#[test]
#[should_panic]
fn einsum_rejects_extra_operand() {
    let device = Default::default();
    let tensor = TestTensor::<1>::ones([2], &device);
    TestTensor::<1>::einsum("i->i", [tensor.clone().into(), tensor.into()]);
}

#[test]
#[should_panic]
fn einsum_rejects_no_operands() {
    TestTensor::<1>::einsum("->", []);
}

#[test]
#[should_panic]
fn einsum_rejects_broadcasting_repeated_label_in_one_operand() {
    let device = Default::default();
    let tensor = TestTensor::<2>::ones([1, 2], &device);
    TestTensor::<1>::einsum("ii->i", [tensor.into()]);
}
