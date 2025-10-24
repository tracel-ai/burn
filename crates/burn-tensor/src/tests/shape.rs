// What this block does:
// Validates `Shape::dims` across truncation, padding, and empty shape cases.
// Why this design:
// Prevents regressions for consumers relying on historical fixed-size array
// representations by covering the primary edge scenarios explicitly.
// Creative touch:
// Curates the assertions as lightweight documentation for backend
// implementers, highlighting nuanced behaviour instead of a single panic test.
#[burn_tensor_testgen::testgen(shape)]
mod tests {
    use super::*;
    use burn_tensor::Shape;

    #[test]
    fn dims_truncates_when_requested_smaller_array() {
        let shape = Shape::new([2, 3, 5, 7]);

        assert_eq!(shape.dims::<2>(), [2, 3]);
    }

    #[test]
    fn dims_pads_with_ones_for_missing_dimensions() {
        let shape = Shape::new([4, 8]);

        assert_eq!(shape.dims::<4>(), [4, 8, 1, 1]);
    }

    #[test]
    fn dims_handles_empty_shapes() {
        let shape = Shape::new([]);

        assert_eq!(shape.dims::<3>(), [1, 1, 1]);
    }
}
