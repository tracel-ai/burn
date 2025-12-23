use super::*;
use burn_tensor::{Element, Shape};

#[test]
fn should_support_float_dtype() {
    let tensor = TestTensor::<2>::from([[0.0, -1.0, 2.0], [3.0, 4.0, -5.0]]).into_primitive();

    assert_eq!(
        burn_tensor::TensorMetadata::shape(&tensor),
        Shape::new([2, 3])
    );
    assert_eq!(
        burn_tensor::TensorMetadata::dtype(&tensor),
        FloatElem::dtype() // default float elem type
    );
}
