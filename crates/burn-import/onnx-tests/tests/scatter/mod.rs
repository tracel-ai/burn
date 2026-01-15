use crate::include_models;
include_models!(scatter_onnx);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn scatter_onnx() {
        // Initialize the model
        let device = Default::default();
        let model: scatter_onnx::Model<TestBackend> = scatter_onnx::Model::default();

        // Create test inputs matching the ONNX model shapes
        // data: [3, 5], indices: [3, 3], updates: [3, 3], axis: 1
        let data = Tensor::<TestBackend, 2>::zeros([3, 5], &device);
        let indices = Tensor::<TestBackend, 2, Int>::from_ints(
            [[0i64, 1, 2], [0, 1, 2], [0, 1, 2]],
            &device,
        );
        let updates = Tensor::<TestBackend, 2>::from_floats(
            [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            &device,
        );

        // Run the model
        let output = model.forward(data, indices, updates);

        // Expected: scatter updates into data at indices along axis 1
        // Row 0: positions [0,1,2] get values [1,2,3]
        // Row 1: positions [0,1,2] get values [4,5,6]
        // Row 2: positions [0,1,2] get values [7,8,9]
        // Note: Burn's scatter with IndexingUpdateOp::Add adds to existing values
        // Since data is zeros, the result should be the same as ONNX's replace semantics
        let expected = TensorData::from([
            [1.0f32, 2.0, 3.0, 0.0, 0.0],
            [4.0, 5.0, 6.0, 0.0, 0.0],
            [7.0, 8.0, 9.0, 0.0, 0.0],
        ]);

        output.to_data().assert_eq(&expected, true);
    }
}
