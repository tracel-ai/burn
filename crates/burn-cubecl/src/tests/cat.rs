#[burn_tensor_testgen::testgen(cat)]
mod tests {
    use super::*;
    use burn_tensor::{backend::Backend, Distribution, Tensor};

    #[test]
    fn cat_should_match_reference_backend_dim0() {
        test_same_as_reference([6, 256], 2, 0);
    }

    #[test]
    fn cat_should_match_reference_backend_dim1() {
        test_same_as_reference([6, 256], 2, 1);
    }

    #[test]
    fn cat_should_support_uneven_launch() {
        test_same_as_reference([1, 137], 2, 0);
    }

    fn test_same_as_reference(shape: [usize; 2], num_tensors: usize, dim: usize) {
        TestBackend::seed(0);
        let tensors = (0..num_tensors)
            .map(|_| {
                Tensor::<TestBackend, 2>::random(shape, Distribution::Default, &Default::default())
            })
            .collect::<Vec<_>>();
        let tensors_ref = tensors
            .iter()
            .map(|tensor| {
                Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default())
            })
            .collect::<Vec<_>>();

        let tensor = Tensor::<TestBackend, 2>::cat(tensors, dim);
        let tensor_ref = Tensor::<ReferenceBackend, 2>::cat(tensors_ref, dim);

        tensor
            .into_data()
            .assert_approx_eq(&tensor_ref.into_data(), 3);
    }
}
