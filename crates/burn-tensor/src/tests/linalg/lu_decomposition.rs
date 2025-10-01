#[burn_tensor_testgen::testgen(lu_decomposition)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, cast::ToElement, linalg::lu_decomposition, Distribution, Shape, s};

    #[test]
    fn test_lu_2x2_decomposition() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data([[4.0, 3.0], [6.0, 3.0]], &device);
        let (result, permutations) = lu_decomposition(tensor).unwrap();
        let expected = TestTensor::<2>::from_data([[6.0, 3.0], [2.0 / 3.0, 1.0]], &device);
        assert_eq!(
            result.to_data(),
            expected.to_data(),
            "LU decomposition result does not match expected value"
        );
    }

    #[test]
    fn test_lu_3x3_decomposition() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data(
            [[0.0, 5.0, 22.0 / 3.0], [4.0, 2.0, 1.0], [2.0, 7.0, 9.0]],
            &device,
        );
        let (result, permutations) = lu_decomposition(tensor).unwrap();
        let expected = TestTensor::<2>::from_data(
            [
                [4.0, 2.0, 1.0],
                [0.5, 6.0, 8.5],
                [0.0, 0.8333333, 0.25000048],
            ],
            &device,
        );
        assert_eq!(
            permutations.to_data().to_vec::<i64>().unwrap(),
            vec![1, 2, 0],
            "LU decomposition permutations do not match expected value"
        );
        assert_eq!(
            result.to_data(),
            expected.to_data(),
            "LU decomposition result does not match expected value"
        );
    }

    #[test]
    fn test_lu_singular_matrix() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data([[1.0, 2.0], [2.0, 4.0]], &device);
        let result = lu_decomposition(tensor);
        assert!(
            result.is_err(),
            "LU decomposition should fail for singular matrices"
        );
    }

    #[test]
    fn test_lu_non_square_matrix() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        let result = lu_decomposition(tensor);
        assert!(
            result.is_err(),
            "LU decomposition should fail for non-square matrices"
        );
    }

    #[test]
    fn test_lu_1x1_element_matrix() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_data([[5.0]], &device);
        let (result, permutations) = lu_decomposition(tensor).unwrap();
        let expected = TestTensor::<2>::from_data([[5.0]], &device);
        assert_eq!(
            result.to_data(),
            expected.to_data(),
            "LU decomposition result does not match expected value for 1x1 matrix"
        );
    }

    #[test]
    fn test_lu_identity_matrix() {
        let device = Default::default();
       
        let tensor = TestTensor::<2>::eye(4, &device);
        let (result, permutations) = lu_decomposition(tensor).unwrap();
        let expected = TestTensor::<2>::eye(4, &device);
        assert_eq!(
            result.to_data(),
            expected.to_data(),
            "LU decomposition result does not match expected value for huge matrix"
        );
    }

    #[test]
    fn test_lu_50x50_random_matrix() {
        let device = Default::default();
        let size = 50;
        let distribution = Distribution::Uniform(0.0, 1.0);
        let tensor = TestTensor::<2>::random(Shape::new([size, size]), distribution, &device);
        let (result, permutations) = lu_decomposition(tensor.clone()).unwrap();
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
            p = p.slice_assign(s![perm_index, i], TestTensor::<2>::from_data([[1.0]], &device));
        }

        // Verify that P * L * U reconstructs the original matrix
        let reconstructed = p.matmul(l).matmul(u);
        let abs_difference = (reconstructed - tensor).abs();
        let max_difference = abs_difference.max();
        let tolerance = 1e-6;
        assert!(
            max_difference.clone().into_scalar().to_f32() < tolerance,
            "Random LU decomposition reconstruction error exceeds tolerance: {} >= {}",
            max_difference.into_scalar().to_f32(),
            tolerance
        );
    }

}
