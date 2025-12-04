use super::*;

#[test]
fn test_eye_float() {
    let device = Default::default();
    let tensor = TestTensor::<2>::from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let rhs = TestTensor::<2>::eye(3, &device);
    assert_eq!(tensor.to_data(), rhs.to_data());
}

#[test]
fn test_eye_int() {
    let device = Default::default();
    let tensor = TestTensorInt::<2>::from([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    let rhs = TestTensorInt::<2>::eye(3, &device);
    assert_eq!(tensor.to_data(), rhs.to_data());
}
