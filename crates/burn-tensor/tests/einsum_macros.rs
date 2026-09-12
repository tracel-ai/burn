//! Public macro integration: binding hygiene, rank inference, and numerical runtime parity.
//!
//! Equation diagnostics and the shape-independent expansion are tested by burn-derive.
//! These regressions also exercise shape-dependent branches in the generated lowering.

use burn_tensor::{Tensor, einsum};
use core::cell::Cell;

fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor.into_data().try_to_vec().unwrap()
}

#[test]
fn generated_bindings_do_not_capture_operand_identifiers() {
    let device = Default::default();
    let __einsum_operand_0 = Tensor::<1>::from_floats([1., 2.], &device);
    let __einsum_operand_1 = Tensor::<1>::from_floats([3., 4.], &device);

    // The first generated binding has the name referenced by the second argument.
    let result = einsum!("i,i->i", __einsum_operand_1, __einsum_operand_0);
    assert_eq!(values(result), [3., 8.]);
}

#[test]
fn operands_are_evaluated_once_in_argument_order() {
    let device = Default::default();
    let calls = Cell::new(0);
    let result = einsum!(
        "i,i,i->",
        {
            assert_eq!(calls.replace(1), 0);
            Tensor::<1>::from_floats([1., 2.], &device)
        },
        {
            assert_eq!(calls.replace(2), 1);
            Tensor::<1>::from_floats([3., 4.], &device)
        },
        {
            assert_eq!(calls.replace(3), 2);
            Tensor::<1>::from_floats([5., 6.], &device)
        },
    );
    assert_eq!(calls.get(), 3);
    assert_eq!(values(result), [63.]);
}

#[test]
fn borrowed_operands_of_different_ranks_remain_available() {
    fn contract(matrix: &Tensor<2>, vector: &Tensor<1>) -> Tensor<1> {
        einsum!("ij,j->i", matrix, vector)
    }

    let device = Default::default();
    let matrix = Tensor::<2>::from_floats([[1., 2.], [3., 4.]], &device);
    let vector = Tensor::<1>::from_floats([10., 20.], &device);

    assert_eq!(values(contract(&matrix, &vector)), [50., 110.]);
    assert_eq!(values(matrix), [1., 2., 3., 4.]);
    assert_eq!(values(vector), [10., 20.]);
}

#[test]
fn implicit_scalar_output_infers_rank_one() {
    let device = Default::default();
    let left = Tensor::<1>::from_floats([1., 2., 3.], &device);
    let right = Tensor::<1>::from_floats([4., 5., 6.], &device);
    let result = einsum!("i,i", left, right);
    assert_eq!(result.dims(), [1]);
    assert_eq!(values(result), [32.]);

    let scalar = Tensor::<1>::from_floats([7.], &device);
    let result = einsum!("", &scalar);
    assert_eq!(result.dims(), [1]);
    assert_eq!(values(result), [7.]);
}

#[test]
fn ellipsis_output_uses_the_contextual_rank() {
    let device = Default::default();
    let input = Tensor::<3>::from_floats([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], &device);
    let scales = Tensor::<1>::from_floats([10., 100.], &device);
    let result: Tensor<3> = einsum!("...i,i->...i", input, scales);
    assert_eq!(result.dims(), [2, 2, 2]);
    assert_eq!(values(result), [10., 200., 30., 400., 50., 600., 70., 800.]);
}

#[test]
fn ellipsis_can_infer_a_generic_output_rank_from_the_return_type() {
    fn identity<const D: usize>(tensor: Tensor<D>) -> Tensor<D> {
        einsum!("...->...", tensor)
    }

    let tensor = Tensor::<2>::from_floats([[1., 2.], [3., 4.]], &Default::default());
    let result = identity(tensor);
    assert_eq!(result.dims(), [2, 2]);
    assert_eq!(values(result), [1., 2., 3., 4.]);
}

#[test]
fn output_ellipsis_without_input_ellipsis_infers_exact_rank() {
    let matrix = Tensor::<2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &Default::default());
    let result = einsum!("ij->...ji", matrix);
    assert_eq!(result.dims(), [3, 2]);
    assert_eq!(values(result), [1., 4., 2., 5., 3., 6.]);
}

#[test]
fn macro_and_runtime_equation_have_identical_results() {
    let device = Default::default();
    let matrix = Tensor::<2>::from_floats([[1., 2.], [3., 4.]], &device);
    let weights = Tensor::<1>::from_floats([10., 20.], &device);
    let projection = Tensor::<2>::from_floats([[1., 2.], [3., 4.]], &device);

    let compiled = einsum!("ij,j,jk->ik", &matrix, &weights, &projection);
    let runtime = Tensor::<2>::einsum(
        "ij,j,jk->ik",
        [(&matrix).into(), (&weights).into(), (&projection).into()],
    );
    assert_eq!(compiled.dims(), runtime.dims());
    assert_eq!(values(compiled), values(runtime));
}

#[test]
fn macro_resolves_tensor_and_execution_through_its_crate() {
    use burn_tensor as renamed;

    struct Tensor;
    struct Execution;
    let _ = (Tensor, Execution);
    let input = renamed::Tensor::<1>::from_floats([2., 3.], &Default::default());
    let result = renamed::einsum!("i->", input);
    assert_eq!(values(result), [5.]);
}

fn assert_einsum_result<const D: usize>(
    compiled: Tensor<D>,
    runtime: Tensor<D>,
    shape: [usize; D],
    expected: &[f32],
) {
    for result in [compiled, runtime] {
        assert_eq!(result.dims(), shape);
        assert_eq!(values(result), expected);
    }
}

#[test]
fn macro_broadcasts_crossed_singleton_batch_axes() {
    let device = Default::default();
    let left = Tensor::<4>::from_floats(
        [
            [[[1., 2., 3.], [4., 5., 6.]]],
            [[[7., 8., 9.], [10., 11., 12.]]],
        ],
        &device,
    );
    let right = Tensor::<4>::from_floats(
        [[
            [[1., 0.], [0., 1.], [1., 1.]],
            [[2., 0.], [0., 2.], [2., 2.]],
            [[0., 1.], [1., 0.], [1., 1.]],
            [[1., 1.], [1., 1.], [1., 1.]],
        ]],
        &device,
    );

    // The shared batch dimensions broadcast in opposite directions. Flattening
    // them before resolving broadcasting would merge incompatible batch sizes.
    assert_einsum_result(
        einsum!("abik,abkj->abij", &left, &right),
        Tensor::einsum("abik,abkj->abij", [(&left).into(), (&right).into()]),
        [2, 4, 2, 2],
        &[
            4., 5., 10., 11., 8., 10., 20., 22., 5., 4., 11., 10., 6., 6., 15., 15., 16., 17., 22.,
            23., 32., 34., 44., 46., 17., 16., 23., 22., 24., 24., 33., 33.,
        ],
    );
}

#[test]
fn macro_broadcasts_contracted_axis_on_either_operand() {
    let device = Default::default();
    let left = Tensor::<2>::from_floats([[2.], [3.]], &device);
    let right = Tensor::<2>::from_floats([[1., 2.], [3., 4.], [5., 6.]], &device);
    let expected = [18., 24., 27., 36.];

    assert_einsum_result(
        einsum!("ik,kj->ij", &left, &right),
        Tensor::einsum("ik,kj->ij", [(&left).into(), (&right).into()]),
        [2, 2],
        &expected,
    );
    assert_einsum_result(
        einsum!("kj,ik->ij", &right, &left),
        Tensor::einsum("kj,ik->ij", [(&right).into(), (&left).into()]),
        [2, 2],
        &expected,
    );
}

#[test]
fn macro_preserves_broadcast_axes_needed_by_later_operands() {
    let device = Default::default();
    let first = Tensor::<1>::from_floats([1., 2.], &device);
    let singleton = Tensor::<1>::from_floats([3.], &device);
    let last = Tensor::<1>::from_floats([4., 5.], &device);

    assert_einsum_result(
        einsum!("i,i,i->", &first, &singleton, &last),
        Tensor::einsum(
            "i,i,i->",
            [(&first).into(), (&singleton).into(), (&last).into()],
        ),
        [1],
        &[42.],
    );

    // A non-singleton contracted dimension may first appear in the last operand.
    let first = Tensor::<1>::from_floats([2.], &device);
    assert_einsum_result(
        einsum!("i,i,i->", &first, &singleton, &last),
        Tensor::einsum(
            "i,i,i->",
            [(&first).into(), (&singleton).into(), (&last).into()],
        ),
        [1],
        &[54.],
    );
}

#[test]
fn macro_restores_output_order_after_contraction() {
    let device = Default::default();
    let left = Tensor::<2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
    let right = Tensor::<2>::from_floats(
        [[1., 2., 3., 4.], [2., 3., 4., 5.], [3., 4., 5., 6.]],
        &device,
    );

    assert_einsum_result(
        einsum!("ik,kj->ji", &left, &right),
        Tensor::einsum("ik,kj->ji", [(&left).into(), (&right).into()]),
        [4, 2],
        &[14., 32., 20., 47., 26., 62., 32., 77.],
    );
}

#[test]
fn macro_handles_empty_contraction_and_output() {
    let device = Default::default();
    let left = Tensor::<2>::zeros([2, 0], &device);
    let right = Tensor::<2>::zeros([0, 3], &device);
    assert_einsum_result(
        einsum!("ik,kj->ij", &left, &right),
        Tensor::einsum("ik,kj->ij", [(&left).into(), (&right).into()]),
        [2, 3],
        &[0.; 6],
    );

    let left = Tensor::<2>::zeros([0, 3], &device);
    let right = Tensor::<1>::from_floats([1., 2., 3.], &device);
    assert_einsum_result(
        einsum!("ij,j->i", &left, &right),
        Tensor::einsum("ij,j->i", [(&left).into(), (&right).into()]),
        [0],
        &[],
    );

    let empty = Tensor::<1>::zeros([0], &device);
    let singleton = Tensor::<1>::from_floats([3.], &device);
    assert_einsum_result(
        einsum!("i,i,i->", &empty, &singleton, &empty),
        Tensor::einsum(
            "i,i,i->",
            [(&empty).into(), (&singleton).into(), (&empty).into()],
        ),
        [1],
        &[0.],
    );
}

#[test]
fn macro_extracts_nonadjacent_and_repeated_diagonals() {
    let device = Default::default();
    let input = Tensor::<3>::from_floats(
        [
            [[1., 2.], [3., 4.], [5., 6.]],
            [[7., 8.], [9., 10.], [11., 12.]],
        ],
        &device,
    );
    assert_einsum_result(
        einsum!("iji->j", &input),
        Tensor::einsum("iji->j", [(&input).into()]),
        [3],
        &[9., 13., 17.],
    );

    let input = Tensor::<3>::from_floats([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], &device);
    assert_einsum_result(
        einsum!("iii->i", &input),
        Tensor::einsum("iii->i", [(&input).into()]),
        [2],
        &[1., 8.],
    );

    let weights = Tensor::<1>::from_floats([2., 3.], &device);
    assert_einsum_result(
        einsum!("iij,j->i", &input, &weights),
        Tensor::einsum("iij,j->i", [(&input).into(), (&weights).into()]),
        [2],
        &[8., 38.],
    );
}

#[test]
fn macro_handles_empty_diagonal_and_trace() {
    let input = Tensor::<2>::zeros([0, 0], &Default::default());
    assert_einsum_result(
        einsum!("ii->i", &input),
        Tensor::einsum("ii->i", [(&input).into()]),
        [0],
        &[],
    );
    assert_einsum_result(
        einsum!("ii->", &input),
        Tensor::einsum("ii->", [(&input).into()]),
        [1],
        &[0.],
    );
}

#[test]
fn macro_reorders_and_reduces_ellipsis_axes() {
    let device = Default::default();
    let input = Tensor::<3>::from_floats([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]], &device);
    assert_einsum_result(
        einsum!("i...j->...ji", &input),
        Tensor::einsum("i...j->...ji", [(&input).into()]),
        [2, 2, 2],
        &[1., 5., 2., 6., 3., 7., 4., 8.],
    );

    let weights = Tensor::<2>::from_floats([[2., 3.], [4., 5.]], &device);
    assert_einsum_result(
        einsum!("...i,...i->", &input, &weights),
        Tensor::einsum("...i,...i->", [(&input).into(), (&weights).into()]),
        [1],
        &[136.],
    );

    let input = Tensor::<1>::from_floats([1., 2.], &device);
    assert_einsum_result(
        einsum!("...i->...i", &input),
        Tensor::einsum("...i->...i", [(&input).into()]),
        [2],
        &[1., 2.],
    );

    let empty = Tensor::<3>::zeros([2, 0, 4], &device);
    assert_einsum_result(
        einsum!("...i->...", &empty),
        Tensor::einsum("...i->...", [(&empty).into()]),
        [2, 0],
        &[],
    );
}

#[test]
#[should_panic(expected = "repeated subscripts must have equal dimensions")]
fn macro_rejects_broadcasting_within_a_diagonal() {
    let input = Tensor::<2>::ones([1, 2], &Default::default());
    let _ = einsum!("ii->i", input);
}

#[cfg(feature = "autodiff")]
#[test]
fn macro_and_runtime_empty_results_preserve_both_gradient_connections() {
    let device = Default::default();
    for (left_shape, right_shape) in [([2, 0], [0, 3]), ([0, 2], [2, 3])] {
        for compiled in [true, false] {
            // Use a fresh graph for each backward pass.
            let left = Tensor::<2>::zeros(left_shape, &device)
                .autodiff()
                .require_grad();
            let right = Tensor::<2>::zeros(right_shape, &device)
                .autodiff()
                .require_grad();
            let output = if compiled {
                einsum!("ik,kj->ij", &left, &right)
            } else {
                Tensor::<2>::einsum("ik,kj->ij", [(&left).into(), (&right).into()])
            };
            let gradients = output.backward();
            for (tensor, shape) in [(&left, left_shape), (&right, right_shape)] {
                let gradient = tensor
                    .grad(&gradients)
                    .expect("operand must remain connected");
                assert_eq!(gradient.dims(), shape);
                assert_eq!(values(gradient), vec![0.; shape.iter().product()]);
            }
        }
    }
}

#[test]
fn macro_extracts_diagonals_across_symbolic_ellipsis_axes() {
    let device = Default::default();
    let data: [[[[f32; 2]; 3]; 2]; 2] = core::array::from_fn(|first| {
        core::array::from_fn(|batch| {
            core::array::from_fn(|column| {
                core::array::from_fn(|last| (first * 100 + batch * 10 + column * 2 + last) as f32)
            })
        })
    });
    let input = Tensor::<4>::from_floats(data, &device);
    assert_einsum_result(
        einsum!("i...i->...", &input),
        Tensor::einsum("i...i->...", [(&input).into()]),
        [2, 3],
        &[101., 105., 109., 121., 125., 129.],
    );

    // With no ellipsis axes, the same symbolic diagonal becomes a matrix trace.
    let input = Tensor::<2>::from_floats([[1., 2.], [3., 4.]], &device);
    assert_einsum_result(
        einsum!("i...i->...", &input),
        Tensor::einsum("i...i->...", [(&input).into()]),
        [1],
        &[5.],
    );

    let data: [[[[f32; 3]; 2]; 2]; 2] = core::array::from_fn(|first| {
        core::array::from_fn(|batch| {
            core::array::from_fn(|second| {
                core::array::from_fn(|column| (first * 12 + batch * 6 + second * 3 + column) as f32)
            })
        })
    });
    let input = Tensor::<4>::from_floats(data, &device);
    assert_einsum_result(
        einsum!("i...ij->j...", &input),
        Tensor::einsum("i...ij->j...", [(&input).into()]),
        [3, 2],
        &[15., 27., 17., 29., 19., 31.],
    );
}

#[test]
fn macro_contracts_before_trailing_singleton_weights_when_possible() {
    let device = Default::default();
    let left = Tensor::<2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
    let right = Tensor::<2>::from_floats([[1., 2.], [3., 4.], [5., 6.]], &device);
    for (weights, expected) in [
        (
            Tensor::<1>::from_floats([3.], &device),
            [66., 84., 147., 192.],
        ),
        (
            Tensor::<1>::from_floats([2., 3., 4.], &device),
            [80., 100., 173., 220.],
        ),
    ] {
        assert_einsum_result(
            einsum!("ij,jk,j->ik", &left, &right, &weights),
            Tensor::einsum(
                "ij,jk,j->ik",
                [(&left).into(), (&right).into(), (&weights).into()],
            ),
            [2, 2],
            &expected,
        );
    }
}
