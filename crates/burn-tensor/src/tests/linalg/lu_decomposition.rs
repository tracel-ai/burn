#[burn_tensor_testgen::testgen(lu_decomposition)]
mod tests {
    use super::*;
    use burn_tensor::{Tensor, linalg::lu_decomposition};

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
}
