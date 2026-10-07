//! Tests that a fused `select` reads its input at the coordinates of its own output.

use super::*;
use burn_tensor::TensorData;

/// A `select` followed by a wider element-wise op in the same block: the wider op must not
/// widen the block's reference past the `select`, whose layout it computes its coordinates from.
#[test]
fn select_then_wider_elementwise_reads_the_selected_index() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();
        let tensor = TestTensorInt::<2>::from_data([[10, 20, 30, 40]], &device).add_scalar(1);
        let indices = TestTensorInt::<1>::from_data([2], &device);
        device.sync().unwrap();

        let selected = tensor.clone().select(1, indices);
        let doubled = tensor.mul_scalar(2);
        let out = TestTensorInt::cat(vec![selected, doubled], 1).into_data();

        assert_eq!(
            out,
            TensorData::from([[31, 22, 42, 62, 82]]).convert_dtype(out.dtype())
        );
    });
}
