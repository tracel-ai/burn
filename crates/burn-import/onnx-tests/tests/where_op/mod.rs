// Import the shared macro
use crate::include_models;
include_models!(
    where_op,
    where_op_broadcast,
    where_op_scalar_x,
    where_op_scalar_y,
    where_op_all_scalar,
    where_shape_all_shapes,
    where_shape_scalar_cond,
    where_shapes_from_inputs,
    where_static_shape
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn where_op() {
        let device = Default::default();
        let model: where_op::Model<TestBackend> = where_op::Model::new(&device);

        let x = Tensor::ones([2, 2], &device);
        let y = Tensor::zeros([2, 2], &device);
        let mask = Tensor::from_bool([[true, false], [false, true]].into(), &device);

        let output = model.forward(mask, x, y);
        let expected = TensorData::from([[1f32, 0.0], [0.0, 1.0]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn where_op_broadcast() {
        let device = Default::default();
        let model: where_op_broadcast::Model<TestBackend> = where_op_broadcast::Model::new(&device);

        let x = Tensor::ones([2], &device);
        let y = Tensor::zeros([2], &device);
        let mask = Tensor::from_bool([[true, false], [false, true]].into(), &device);

        let output = model.forward(mask, x, y);
        let expected = TensorData::from([[1f32, 0.0], [0.0, 1.0]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn where_op_scalar_x() {
        let device = Default::default();
        let model: where_op_scalar_x::Model<TestBackend> = where_op_scalar_x::Model::new(&device);

        let x = 1.0f32;
        let y = Tensor::zeros([2, 2], &device);
        let mask = Tensor::from_bool([[true, false], [false, true]].into(), &device);

        let output = model.forward(mask, x, y);
        let expected = TensorData::from([[1f32, 0.0], [0.0, 1.0]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn where_op_scalar_y() {
        let device = Default::default();
        let model: where_op_scalar_y::Model<TestBackend> = where_op_scalar_y::Model::new(&device);

        let x = Tensor::ones([2, 2], &device);
        let y = 0.0f32;
        let mask = Tensor::from_bool([[true, false], [false, true]].into(), &device);

        let output = model.forward(mask, x, y);
        let expected = TensorData::from([[1f32, 0.0], [0.0, 1.0]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn where_op_all_scalar() {
        let device = Default::default();
        let model: where_op_all_scalar::Model<TestBackend> =
            where_op_all_scalar::Model::new(&device);

        let x = 1.0f32;
        let y = 0.0f32;
        let mask = true;

        let output = model.forward(mask, x, y);
        let expected = 1.0f32;

        assert_eq!(output, expected);
    }

    #[test]
    fn where_shape_all_shapes() {
        let device = Default::default();
        let model: where_shape_all_shapes::Model<TestBackend> =
            where_shape_all_shapes::Model::new(&device);

        // Create input tensors with specific shapes
        let input1 = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        let input2 = Tensor::<TestBackend, 3>::ones([5, 6, 7], &device);
        let input3 = Tensor::<TestBackend, 3>::ones([10, 20, 30], &device);
        let input4 = Tensor::<TestBackend, 3>::ones([100, 200, 300], &device);

        let output = model.forward(input1, input2, input3, input4);
        // Since shapes are different, Equal will return [0, 0, 0]
        // Where will select all from shape_y [100, 200, 300]
        let expected: [i64; 3] = [100, 200, 300];

        assert_eq!(output, expected);
    }

    #[test]
    fn where_shape_scalar_cond() {
        let device = Default::default();
        let model: where_shape_scalar_cond::Model<TestBackend> =
            where_shape_scalar_cond::Model::new(&device);

        // Create input tensors
        let input1 = Tensor::<TestBackend, 3>::ones([2, 3, 4], &device);
        let input2 = Tensor::<TestBackend, 3>::ones([5, 6, 7], &device);

        // Test with true condition
        let condition = true;
        let output = model.forward(condition, input1.clone(), input2.clone());
        let expected: [i64; 3] = [2, 3, 4]; // Should select shape1
        assert_eq!(output, expected);

        // Test with false condition
        let condition = false;
        let output = model.forward(condition, input1, input2);
        let expected: [i64; 3] = [5, 6, 7]; // Should select shape2
        assert_eq!(output, expected);
    }

    #[test]
    fn where_shapes_from_inputs() {
        let device = Default::default();
        let model: where_shapes_from_inputs::Model<TestBackend> =
            where_shapes_from_inputs::Model::new(&device);

        // Create input tensors with shapes [1,2,3], [4,5,6], [7,8,9], and [1,0,3]
        let input1 = Tensor::<TestBackend, 3>::ones([1, 2, 3], &device);
        let input2 = Tensor::<TestBackend, 3>::ones([4, 5, 6], &device);
        let input3 = Tensor::<TestBackend, 3>::ones([7, 8, 9], &device);
        let input4 = Tensor::<TestBackend, 3>::ones([1, 0, 3], &device); // Note: The shape matters, not the content

        let output = model.forward(input1, input2, input3, input4);
        // Condition is shape1 == shape4 -> [1, 0, 1] (true=1, false=0)
        // So output should be [shape2[0], shape3[1], shape2[2]] = [4, 8, 6]
        let expected: [i64; 3] = [4, 8, 6];

        assert_eq!(output, expected);
    }

    #[test]
    fn where_static_shape() {
        let device = Default::default();
        // Use Model::default() to load constants from the record file
        let model: where_static_shape::Model<TestBackend> = where_static_shape::Model::default();

        // Create condition tensor (needs to be Bool type)
        let condition = Tensor::from_bool([[true, false], [false, true]].into(), &device);

        let output = model.forward(condition);

        // The model has constant tensors [[1,2],[3,4]] and [[5,6],[7,8]]
        // With condition [[true,false],[false,true]] the output should be [[1,6],[7,4]]
        let expected = TensorData::from([[1.0f32, 6.0], [7.0, 4.0]]);

        output.to_data().assert_eq(&expected, true);
    }
}
