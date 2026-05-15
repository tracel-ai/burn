use super::*;
use burn_tensor::{Element, Shape};

#[test]
fn should_support_int_dtype() {
    let tensor = TestTensorInt::<2>::from([[0, -1, 2], [3, 4, -5]])/*.into_primitive()*/;

    assert_eq!(tensor.shape(), Shape::new([2, 3]));
    assert_eq!(
        tensor.dtype(),
        IntElem::dtype() // default int elem type
    );
}
