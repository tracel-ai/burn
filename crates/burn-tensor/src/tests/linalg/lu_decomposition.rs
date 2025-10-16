#[burn_tensor_testgen::testgen(lu_decomposition)]
mod tests {
    use super::*;
    use burn_tensor::ops::FloatElem;
    use burn_tensor::{
        Distribution, Int, Shape, Tensor, TensorData, Tolerance, cast::ToElement,
        linalg::lu_decomposition, s,
    };

    #[test]
    fn test_lu_2x2_decomposition() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data([[4.0, 3.0], [6.0, 3.0]], &device);
        let (result, permutations) = lu_decomposition(tensor);
        let expected = TestTensor::<2>::from_data([[6.0, 3.0], [2.0 / 3.0, 1.0]], &device);
        result.into_data().assert_eq(&expected.into_data(), true);
    }

    #[test]
    fn test_lu_3x3_decomposition() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data(
            [[0.0, 5.0, 22.0 / 3.0], [4.0, 2.0, 1.0], [2.0, 7.0, 9.0]],
            &device,
        );
        let (result, permutations) = lu_decomposition(tensor);
        let expected = TestTensor::<2>::from_data(
            [
                [4.0, 2.0, 1.0],
                [0.5, 6.0, 8.5],
                [0.0, 0.8333333, 0.25000048],
            ],
            &device,
        );
        let expected_permutations =
            Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([1, 2, 0]), &device);
        permutations
            .into_data()
            .assert_eq(&expected_permutations.into_data(), true);
        result
            .into_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(
                &expected.into_data(),
                Tolerance::default(),
            );
    }

    #[test]
    #[should_panic]
    fn test_lu_singular_matrix() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [2.0, 4.0]], &device);
        let result = lu_decomposition(tensor);
    }

    #[test]
    #[should_panic]
    fn test_lu_non_square_matrix() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        let result = lu_decomposition(tensor);
    }

    #[test]
    fn test_lu_1x1_element_matrix() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data([[5.0]], &device);
        let (result, permutations) = lu_decomposition(tensor);
        let expected = TestTensor::<2>::from_data([[5.0]], &device);

        result.into_data().assert_eq(&expected.into_data(), true);
    }

    #[test]
    fn test_lu_identity_matrix() {
        let device = Default::default();

        let tensor = TestTensor::<2>::eye(4, &device);
        let (result, permutations) = lu_decomposition(tensor);
        let expected = TestTensor::<2>::eye(4, &device);
        result.into_data().assert_eq(&expected.into_data(), true);
    }

    #[test]
    fn test_lu_50x50_random_matrix() {
        let device = Default::default();
        let size = 50;
        let distribution = Distribution::Uniform(0.0, 1.0);
        let tensor = TestTensor::<2>::random(Shape::new([size, size]), distribution, &device);
        let (result, permutations) = lu_decomposition(tensor.clone());
        // Reconstruct the original matrix from L and U
        let mut l = TestTensor::<2>::eye(size, &device);
        let mut u = TestTensor::<2>::zeros(Shape::new([size, size]), &device);

        for i in 0..size {
            for j in 0..size {
                if i > j {
                    l = l.slice_assign(s![i, j], result.clone().slice(s![i, j]));
                } else {
                    u = u.slice_assign(s![i, j], result.clone().slice(s![i, j]));
                }
            }
        }
        // Construct the permutation matrix P from the permutation vector
        let mut p = TestTensor::<2>::zeros(Shape::new([size, size]), &device);
        for i in 0..size {
            let perm_index = permutations.clone().slice(s![i]).into_scalar().to_usize();
            p = p.slice_assign(
                s![perm_index, i],
                TestTensor::<2>::from_data([[1.0]], &device),
            );
        }

        // Verify that P * L * U reconstructs the original matrix
        let reconstructed = p.matmul(l).matmul(u);
        reconstructed
            .into_data()
            .assert_approx_eq::<FloatElem<TestBackend>>(&tensor.into_data(), Tolerance::default());
    }
}
