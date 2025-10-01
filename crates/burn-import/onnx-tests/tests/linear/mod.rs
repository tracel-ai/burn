use crate::include_models;
include_models!(linear);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use float_cmp::ApproxEq;

    use crate::backend::TestBackend;

    #[test]
    fn linear() {
        let device = Default::default();
        // Initialize the model with weights (loaded from the exported file)
        let model: linear::Model<TestBackend> = linear::Model::default();
        #[allow(clippy::approx_constant)]
        let input1 = Tensor::<TestBackend, 2>::full([4, 3], 3.14, &device);
        #[allow(clippy::approx_constant)]
        let input2 = Tensor::<TestBackend, 2>::full([2, 5], 3.14, &device);
        #[allow(clippy::approx_constant)]
        let input3 = Tensor::<TestBackend, 3>::full([3, 2, 7], 3.14, &device);

        let (output1, output2, output3) = model.forward(input1, input2, input3);

        // test the output shape
        let expected_shape1: Shape = Shape::from([4, 4]);
        let expected_shape2: Shape = Shape::from([2, 6]);
        let expected_shape3: Shape = Shape::from([3, 2, 8]);
        assert_eq!(output1.shape(), expected_shape1);
        assert_eq!(output2.shape(), expected_shape2);
        assert_eq!(output3.shape(), expected_shape3);

        // We are using the sum of the output tensor to test the correctness of the conv1d node
        // because the output tensor is too large to compare with the expected tensor.
        let output_sum1 = output1.sum().into_scalar();
        let output_sum2 = output2.sum().into_scalar();
        let output_sum3 = output3.sum().into_scalar();

        let expected_sum1 = -9.655_477; // from pytorch
        let expected_sum2 = -8.053_822; // from pytorch
        let expected_sum3 = 27.575_281; // from pytorch

        assert!(expected_sum1.approx_eq(output_sum1, (1.0e-5, 2)));
        assert!(expected_sum2.approx_eq(output_sum2, (1.0e-5, 2)));
        assert!(expected_sum3.approx_eq(output_sum3, (1.0e-5, 2)));
    }
}
