#[burn_tensor_testgen::testgen(diagonal)]

mod tests {
    use super::*;
    use burn_tensor::backend::Backend;
    use burn_tensor::{Data, Tensor};

    type FloatElem = <TestBackend as Backend>::FloatElem;
    type IntElem = <TestBackend as Backend>::IntElem;

    #[test]
    fn test_diagonal() {
        let data = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
        let lhs = Tensor::<TestBackend, 2>::from_floats(data);
        let rhs = Tensor::<TestBackend, 2>::diagonal(3);
        lhs.to_data().assert_approx_eq(&rhs.to_data(), 3);
    }
}
