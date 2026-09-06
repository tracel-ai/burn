//! Public macro integration: binding hygiene, borrowing, and output type inference.
//!
//! Equation diagnostics and the shape-independent expansion are tested by burn-derive;
//! backend suites cover the numerical contraction cases.

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
