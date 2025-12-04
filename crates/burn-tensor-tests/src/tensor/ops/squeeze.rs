use crate::*;
use burn_tensor::Shape;

/// Test if the function can successfully squeeze the size 1 dimension of a 3D tensor.
#[test]
fn should_squeeze_dim() {
    let tensor = TestTensor::<3>::ones(Shape::new([2, 1, 4]), &Default::default());
    let squeezed_tensor: TestTensor<2> = tensor.squeeze_dim(1);
    let expected_shape = Shape::new([2, 4]);
    assert_eq!(squeezed_tensor.shape(), expected_shape);
}

#[test]
fn should_squeeze() {
    let tensor = TestTensor::<3>::ones(Shape::new([2, 1, 4]), &Default::default());
    let squeezed_tensor: TestTensor<2> = tensor.squeeze();
    let expected_shape = Shape::new([2, 4]);
    assert_eq!(squeezed_tensor.shape(), expected_shape);
}

/// Test if the function can successfully squeeze the first size 1 dimension of a 4D tensor.
#[test]
fn should_squeeze_first() {
    let tensor = TestTensor::<4>::ones(Shape::new([1, 3, 4, 5]), &Default::default());
    let squeezed_tensor: TestTensor<3> = tensor.squeeze_dim(0);
    let expected_shape = Shape::new([3, 4, 5]);
    assert_eq!(squeezed_tensor.shape(), expected_shape);
}
/// Test if the function can successfully squeeze the last size 1 dimension of a 4D tensor.
#[test]
fn should_squeeze_last() {
    let tensor = TestTensor::<4>::ones(Shape::new([2, 3, 4, 1]), &Default::default());
    let squeezed_tensor: TestTensor<3> = tensor.squeeze_dim(3);
    let expected_shape = Shape::new([2, 3, 4]);
    assert_eq!(squeezed_tensor.shape(), expected_shape);
}
/// Test if the function panics when the squeezed dimension is not of size 1.
#[test]
#[should_panic]
fn should_squeeze_panic() {
    let tensor = TestTensor::<4>::ones(Shape::new([2, 3, 4, 5]), &Default::default());
    let _squeezed_tensor: TestTensor<3> = tensor.squeeze_dim(2);
}

/// Test if the function works with an empty slice
#[test]
fn should_squeeze_dims_with_empty_slice() {
    let tensor = TestTensor::<3>::ones(Shape::new([1, 1, 3]), &Default::default());
    let squeezed_tensor: TestTensor<1> = tensor.squeeze_dims(&[]);
    let expected_shape = Shape::new([3]);
    assert_eq!(squeezed_tensor.shape(), expected_shape);
}

#[test]
fn should_squeeze_all_dims() {
    let tensor = TestTensor::<3>::ones(Shape::new([1, 3, 1]), &Default::default());
    let squeezed_tensor: TestTensor<1> = tensor.squeeze();
    let expected_shape = Shape::new([3]);
    assert_eq!(squeezed_tensor.shape(), expected_shape);
}

/// Test if the function works with positive indices
#[test]
fn should_squeeze_dims_with_positive_indices() {
    let tensor = TestTensor::<4>::ones(Shape::new([1, 3, 1, 5]), &Default::default());
    let squeezed_tensor: TestTensor<2> = tensor.squeeze_dims(&[0, 2]);
    let expected_shape = Shape::new([3, 5]);
    assert_eq!(squeezed_tensor.shape(), expected_shape);
}

/// Test if the function works with negative indices
#[test]
fn should_squeeze_dims_with_negative_indices() {
    let tensor = TestTensor::<4>::ones(Shape::new([2, 1, 3, 1]), &Default::default());
    let squeezed_tensor: TestTensor<2> = tensor.squeeze_dims(&[-3, -1]);
    let expected_shape = Shape::new([2, 3]);
    assert_eq!(squeezed_tensor.shape(), expected_shape);
}

/// Test to make sure the function panics if a non-singleton dimension is squeezed
#[test]
#[should_panic]
fn should_squeeze_dims_work_if_non_singleton() {
    let tensor = TestTensor::<3>::ones(Shape::new([2, 3, 4]), &Default::default());
    let squeezed_tensor: TestTensor<3> = tensor.squeeze_dims(&[1]);
    let expected_shape = Shape::new([2, 3, 4]);
    assert_eq!(squeezed_tensor.shape(), expected_shape);
}

#[test]
#[should_panic]
fn should_panic_squeeze_consumes_all_singleton() {
    let tensor = TestTensor::<3>::ones(Shape::new([1, 3, 1]), &Default::default());
    let _squeezed_tensor: TestTensor<2> = tensor.squeeze(); // output rank should be 1
}

/// Test to make sure the function panics if too many dimensions are requested to be squeezed
#[test]
#[should_panic]
fn should_squeeze_dims_panic_on_too_many_dimensions() {
    let tensor = TestTensor::<3>::ones(Shape::new([1, 1, 1]), &Default::default());
    let _: TestTensor<1> = tensor.squeeze_dims(&[0, 1, 2]);
}

/// Test to make sure function panics if dimensions are mismatched
#[test]
#[should_panic]
fn should_squeeze_dims_dimension_mismatch_panic() {
    let tensor = TestTensor::<4>::ones(Shape::new([1, 3, 1, 5]), &Default::default());
    let _: TestTensor<3> = tensor.squeeze_dims(&[0, 2]);
}

/// Test if the function can successfully unsqueeze the size 1 dimension at the specified position of a 3D tensor.
#[test]
fn should_unsqueeze_dim() {
    let tensor = TestTensor::<3>::ones(Shape::new([2, 4, 1]), &Default::default());
    let unsqueezed_tensor: TestTensor<4> = tensor.unsqueeze_dim(1);
    let expected_shape = Shape::new([2, 1, 4, 1]);
    assert_eq!(unsqueezed_tensor.shape(), expected_shape);
}

/// Test if the function can successfully unsqueeze the first size 1 dimension of a 4D tensor.
#[test]
fn should_unsqueeze_dim_first() {
    let tensor = TestTensor::<4>::ones(Shape::new([2, 3, 4, 5]), &Default::default());
    let unsqueezed_tensor: TestTensor<5> = tensor.unsqueeze_dim(0);
    let expected_shape = Shape::new([1, 2, 3, 4, 5]);
    assert_eq!(unsqueezed_tensor.shape(), expected_shape);
}

/// Test if the function can successfully unsqueeze the last size 1 dimension of a 4D tensor.
#[test]
fn should_unsqueeze_dim_last() {
    let tensor = TestTensor::<4>::ones(Shape::new([5, 4, 3, 2]), &Default::default());
    let unsqueezed_tensor: TestTensor<5> = tensor.unsqueeze_dim(4);
    let expected_shape = Shape::new([5, 4, 3, 2, 1]);
    assert_eq!(unsqueezed_tensor.shape(), expected_shape);
}

/// Test if the function panics when the unsqueezed dimension is out of bounds.
#[test]
#[should_panic]
fn should_unsqueeze_dim_panic() {
    let tensor = TestTensor::<4>::ones(Shape::new([2, 3, 4, 5]), &Default::default());
    let _unsqueezed_tensor: TestTensor<5> = tensor.unsqueeze_dim(5);
}

#[test]
fn should_unsqueeze_dims_support_dim_inference() {
    let input_tensor = TestTensor::<3>::ones(Shape::new([3, 4, 5]), &Default::default());
    let output_tensor = input_tensor.unsqueeze_dims::<5>(&[1, -2]);
    let expected_shape = Shape::new([3, 1, 4, 1, 5]);
    assert_eq!(output_tensor.shape(), expected_shape);
}

#[test]
fn should_unsqueeze_dims_handle_first_last() {
    let input_tensor = TestTensor::<3>::ones(Shape::new([3, 4, 5]), &Default::default());
    let output_tensor = input_tensor.unsqueeze_dims::<5>(&[0, 4]);
    let expected_shape = Shape::new([1, 3, 4, 5, 1]);
    assert_eq!(output_tensor.shape(), expected_shape);
}

#[test]
fn should_unsqueeze_dims_work_with_single_dim() {
    //bruh, just call unsqueeze_dim
    let input_tensor = TestTensor::<3>::ones(Shape::new([3, 4, 5]), &Default::default());
    let output_tensor: TestTensor<4> = input_tensor.unsqueeze_dims(&[1]);
    let expected_shape = Shape::new([3, 1, 4, 5]);
    assert_eq!(output_tensor.shape(), expected_shape);
}

#[test]
fn should_unsqueeze_dims_multiple_trailing_negatives() {
    let input_tensor = TestTensor::<3>::ones(Shape::new([3, 4, 5]), &Default::default());
    let output_tensor: TestTensor<6> = input_tensor.unsqueeze_dims(&[0, -1, -1]);
    let expected_shape = Shape::new([1, 3, 4, 5, 1, 1]);
    assert_eq!(output_tensor.shape(), expected_shape);
}

#[test]
#[should_panic]
fn should_unsqueeze_dims_panic() {
    let input_tensor = TestTensor::<3>::ones(Shape::new([3, 4, 5]), &Default::default());
    let _output_tensor: TestTensor<5> = input_tensor.unsqueeze_dims(&[0, -6]);
}

#[test]
#[should_panic]
fn squeeze_all_singleton_not_supported() {
    let tensor = TestTensor::<3>::ones(Shape::new([1, 1, 1]), &Default::default());
    let _ = tensor.squeeze::<0>();
}

#[test]
#[should_panic]
fn squeeze_dim_singleton_not_supported() {
    let tensor = TestTensor::<1>::ones(Shape::new([1]), &Default::default());
    let _ = tensor.squeeze_dim::<0>(0);
}

#[test]
#[should_panic]
fn squeeze_dims_all_singleton_not_supported() {
    let tensor = TestTensor::<3>::ones(Shape::new([1, 1, 1]), &Default::default());
    let _ = tensor.squeeze_dims::<0>(&[0, 1, 2]);
}
