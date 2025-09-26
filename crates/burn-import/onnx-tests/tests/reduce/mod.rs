use crate::include_models;
include_models!(
    reduce_max,
    reduce_min,
    reduce_mean,
    reduce_prod,
    reduce_sum,
    reduce_sum_square,
    reduce_l1,
    reduce_l2,
    reduce_log_sum,
    reduce_log_sum_exp
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    // Helper function to assert scalar values with tolerance
    fn assert_scalar_approx_eq(actual: f32, expected: f32, tolerance: f64) {
        let diff = (actual as f64 - expected as f64).abs();
        assert!(
            diff < tolerance,
            "Expected {}, got {}, diff: {}",
            expected,
            actual,
            diff
        );
    }

    #[test]
    fn reduce_min() {
        let device = Default::default();
        let model: reduce_min::Model<TestBackend> = reduce_min::Model::new(&device);

        // Run the models
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [1.0, 4.0, 9.0, 25.0], //
                [2.0, 5.0, 10.0, 26.0],
            ]]],
            &device,
        );
        let (output1, output2, output3, output4, output5, output6) = model.forward(input.clone());
        // output1 and output2 are now scalars (f32), not tensors
        let expected1_scalar = 1.0f32;
        let expected2_scalar = 1.0f32;
        let expected3 = input.to_data();
        let expected4 = TensorData::from([[[[1.0f32], [2.]]]]);
        let expected5 = input.clone().squeeze_dim::<3>(0).to_data();
        let expected6 = TensorData::from([[1.0f32, 4., 9., 25.]]);

        // Assert scalar outputs
        assert_scalar_approx_eq(output1, expected1_scalar, 1e-6);
        assert_scalar_approx_eq(output2, expected2_scalar, 1e-6);
        output3.to_data().assert_eq(&expected3, true);
        output4.to_data().assert_eq(&expected4, true);
        output5.to_data().assert_eq(&expected5, true);
        output6.to_data().assert_eq(&expected6, true);
    }

    #[test]
    fn reduce_max() {
        let device = Default::default();
        let model: reduce_max::Model<TestBackend> = reduce_max::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [1.0, 4.0, 9.0, 25.0], //
                [2.0, 5.0, 10.0, 26.0],
            ]]],
            &device,
        );
        let (output1, output2, output3, output4, output5, output6) = model.forward(input.clone());
        // output1 and output2 are now scalars (f32), not tensors
        let expected1_scalar = 26.0f32;
        let expected2_scalar = 26.0f32;
        let expected3 = input.to_data();
        let expected4 = TensorData::from([[[[25.0f32], [26.]]]]);
        let expected5 = input.clone().squeeze_dim::<3>(0).to_data();
        let expected6 = TensorData::from([[2.0f32, 5., 10., 26.]]);

        // Assert scalar outputs
        assert_scalar_approx_eq(output1, expected1_scalar, 1e-6);
        assert_scalar_approx_eq(output2, expected2_scalar, 1e-6);
        output3.to_data().assert_eq(&expected3, true);
        output4.to_data().assert_eq(&expected4, true);
        output5.to_data().assert_eq(&expected5, true);
        output6.to_data().assert_eq(&expected6, true);
    }

    #[test]
    fn reduce_sum() {
        let device = Default::default();
        let model: reduce_sum::Model<TestBackend> = reduce_sum::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [1.0, 4.0, 9.0, 25.0], //
                [2.0, 5.0, 10.0, 26.0],
            ]]],
            &device,
        );
        let (output1, output2, output3, output4, output5, output6) = model.forward(input.clone());
        // output1 and output2 are now scalars (f32), not tensors
        let expected1_scalar = 82.0f32;
        let expected2_scalar = 82.0f32;
        let expected3 = input.to_data();
        let expected4 = TensorData::from([[[[39.0f32], [43.]]]]);
        let expected5 = input.clone().squeeze_dim::<3>(0).to_data();
        let expected6 = TensorData::from([[3.0f32, 9., 19., 51.]]);

        // Assert scalar outputs
        assert_scalar_approx_eq(output1, expected1_scalar, 1e-6);
        assert_scalar_approx_eq(output2, expected2_scalar, 1e-6);
        output3.to_data().assert_eq(&expected3, true);
        output4
            .to_data()
            .assert_approx_eq::<FT>(&expected4, burn::tensor::Tolerance::default());
        output5.to_data().assert_eq(&expected5, true);
        output6
            .to_data()
            .assert_approx_eq::<FT>(&expected6, burn::tensor::Tolerance::default());
    }

    #[test]
    fn reduce_prod() {
        let device = Default::default();
        let model: reduce_prod::Model<TestBackend> = reduce_prod::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [1.0, 4.0, 9.0, 25.0], //
                [2.0, 5.0, 10.0, 26.0],
            ]]],
            &device,
        );
        let (output1, output2, output3, output4) = model.forward(input.clone());
        // output1 is now scalar (f32), others are tensors
        let expected1_scalar = 2340000.0f32;
        let expected2 = input.to_data();
        let expected3 = TensorData::from([[[[900.0f32], [2600.]]]]);
        let expected4 = TensorData::from([[[2.0f32, 20., 90., 650.]]]);

        // Assert scalar output
        assert_scalar_approx_eq(output1, expected1_scalar, 1e-6);
        output2.to_data().assert_eq(&expected2, true);
        output3
            .to_data()
            .assert_approx_eq::<FT>(&expected3, burn::tensor::Tolerance::default());
        output4
            .to_data()
            .assert_approx_eq::<FT>(&expected4, burn::tensor::Tolerance::default());
    }

    #[test]
    fn reduce_mean() {
        let device = Default::default();
        let model: reduce_mean::Model<TestBackend> = reduce_mean::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [1.0, 4.0, 9.0, 25.0], //
                [2.0, 5.0, 10.0, 26.0],
            ]]],
            &device,
        );
        let (output1, output2, output3, output4, output5, output6) = model.forward(input.clone());
        // output1 and output2 are now scalars (f32), not tensors
        let expected1_scalar = 10.25f32;
        let expected2_scalar = 10.25f32;
        let expected3 = input.to_data();
        let expected4 = TensorData::from([[[[9.75f32], [10.75]]]]);
        let expected5 = input.clone().squeeze_dim::<3>(0).to_data();
        let expected6 = TensorData::from([[1.5f32, 4.5, 9.5, 25.5]]);

        // Assert scalar outputs
        assert_scalar_approx_eq(output1, expected1_scalar, 1e-6);
        assert_scalar_approx_eq(output2, expected2_scalar, 1e-6);
        output3
            .to_data()
            .assert_approx_eq::<FT>(&expected3, burn::tensor::Tolerance::default());
        output4
            .to_data()
            .assert_approx_eq::<FT>(&expected4, burn::tensor::Tolerance::default());
        output5
            .to_data()
            .assert_approx_eq::<FT>(&expected5, burn::tensor::Tolerance::default());
        output6
            .to_data()
            .assert_approx_eq::<FT>(&expected6, burn::tensor::Tolerance::default());
    }

    #[test]
    fn reduce_sum_square() {
        let device = Default::default();
        let model: reduce_sum_square::Model<TestBackend> = reduce_sum_square::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [1.0, 4.0, 9.0, 25.0], //
                [2.0, 5.0, 10.0, 26.0],
            ]]],
            &device,
        );
        let (output1, output2, output3, output4, output5) = model.forward(input.clone());

        // output1 is now scalar (f32), others are tensors
        let expected1_scalar = 1528.0f32;
        let expected2 = TensorData::from([[[
            [1.0f32, 16.0, 81.0, 625.0], //
            [4.0, 25.0, 100.0, 676.0],
        ]]]);
        let expected3 = TensorData::from([[[[723.0f32], [805.0]]]]);
        let expected4 = TensorData::from([[
            [1.0f32, 16.0, 81.0, 625.0], //
            [4.0, 25.0, 100.0, 676.0],
        ]]);
        let expected5 = TensorData::from([[5.0f32, 41.0, 181.0, 1301.0]]);

        // Assert scalar output
        assert_scalar_approx_eq(output1, expected1_scalar, 1e-6);
        output2
            .to_data()
            .assert_approx_eq::<FT>(&expected2, burn::tensor::Tolerance::default());
        output3
            .to_data()
            .assert_approx_eq::<FT>(&expected3, burn::tensor::Tolerance::default());
        output4
            .to_data()
            .assert_approx_eq::<FT>(&expected4, burn::tensor::Tolerance::default());
        output5
            .to_data()
            .assert_approx_eq::<FT>(&expected5, burn::tensor::Tolerance::default());
    }

    #[test]
    fn reduce_l1() {
        let device = Default::default();
        let model: reduce_l1::Model<TestBackend> = reduce_l1::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [1.0, -4.0, 9.0, 25.0], //
                [2.0, 5.0, -10.0, 26.0],
            ]]],
            &device,
        );
        // let (output2, output3, output4, output5) = model.forward(input.clone());
        let (output1, output2, output3, output4, output5) = model.forward(input.clone());

        // output1 is now scalar (f32), others are tensors
        let expected1_scalar = 82.0f32;
        let expected2 = TensorData::from([[[
            [1.0f32, 4.0, 9.0, 25.], //
            [2.0, 5.0, 10.0, 26.0],
        ]]]);
        let expected3 = TensorData::from([[[[39.0f32], [43.0]]]]);
        let expected4 = TensorData::from([[
            [1.0f32, 4.0, 9.0, 25.0], //
            [2.0, 5.0, 10.0, 26.0],
        ]]);
        let expected5 = TensorData::from([[3.0f32, 9.0, 19.0, 51.0]]);

        // Assert scalar output
        assert_scalar_approx_eq(output1, expected1_scalar, 1e-6);
        output2
            .to_data()
            .assert_approx_eq::<FT>(&expected2, burn::tensor::Tolerance::default());
        output3
            .to_data()
            .assert_approx_eq::<FT>(&expected3, burn::tensor::Tolerance::default());
        output4
            .to_data()
            .assert_approx_eq::<FT>(&expected4, burn::tensor::Tolerance::default());
        output5
            .to_data()
            .assert_approx_eq::<FT>(&expected5, burn::tensor::Tolerance::default());
    }

    #[test]
    fn reduce_l2() {
        let device = Default::default();
        let model: reduce_l2::Model<TestBackend> = reduce_l2::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [1.0, -4.0, 9.0, 25.0], //
                [2.0, 5.0, -10.0, 26.0],
            ]]],
            &device,
        );
        let (output1, output2, output3, output4, output5) = model.forward(input.clone());

        // output1 is now scalar (f32), others are tensors
        let expected1_scalar = 39.08964057138413f32;
        let expected2 = TensorData::from([[[
            [1.0f32, 4.0, 9.0, 25.], //
            [2.0, 5.0, 10.0, 26.0],
        ]]]);
        let expected3 = TensorData::from([[[[26.88865932f32], [28.37252192]]]]);
        let expected4 = TensorData::from([[
            [1.0f32, 4.0, 9.0, 25.0], //
            [2.0, 5.0, 10.0, 26.0],
        ]]);
        let expected5 = TensorData::from([[2.23606798f32, 6.40312424, 13.45362405, 36.06937759]]);

        // Assert scalar output
        assert_scalar_approx_eq(output1, expected1_scalar, 1e-6);
        output2
            .to_data()
            .assert_approx_eq::<FT>(&expected2, burn::tensor::Tolerance::default());
        output3
            .to_data()
            .assert_approx_eq::<FT>(&expected3, burn::tensor::Tolerance::default());
        output4
            .to_data()
            .assert_approx_eq::<FT>(&expected4, burn::tensor::Tolerance::default());
        output5
            .to_data()
            .assert_approx_eq::<FT>(&expected5, burn::tensor::Tolerance::default());
    }

    #[test]
    fn reduce_log_sum() {
        let device = Default::default();
        let model: reduce_log_sum::Model<TestBackend> = reduce_log_sum::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [1.0, 4.0, 9.0, 25.0], //
                [2.0, 5.0, 10.0, 26.0],
            ]]],
            &device,
        );
        let (output1, output2, output3, output4, output5) = model.forward(input.clone());

        // output1 is now scalar (f32), others are tensors
        let expected1_scalar = 4.406719247264253f32;
        let expected2 = TensorData::from([[[
            [0.0f32, 1.38629436, 2.19722458, 3.21887582], //
            [0.69314718, 1.60943791, 2.30258509, 3.25809654],
        ]]]);
        let expected3 = TensorData::from([[[[3.66356165f32], [3.76120012]]]]);
        let expected4 = TensorData::from([[
            [0.0f32, 1.38629436, 2.19722458, 3.21887582], //
            [0.69314718, 1.60943791, 2.30258509, 3.25809654],
        ]]);
        let expected5 = TensorData::from([[1.09861229f32, 2.19722458, 2.94443898, 3.93182563]]);

        // Assert scalar output
        assert_scalar_approx_eq(output1, expected1_scalar, 1e-6);
        output2
            .to_data()
            .assert_approx_eq::<FT>(&expected2, burn::tensor::Tolerance::default());
        output3
            .to_data()
            .assert_approx_eq::<FT>(&expected3, burn::tensor::Tolerance::default());
        output4
            .to_data()
            .assert_approx_eq::<FT>(&expected4, burn::tensor::Tolerance::default());
        output5
            .to_data()
            .assert_approx_eq::<FT>(&expected5, burn::tensor::Tolerance::default());
    }

    #[test]
    fn reduce_log_sum_exp() {
        let device = Default::default();
        let model: reduce_log_sum_exp::Model<TestBackend> = reduce_log_sum_exp::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [1.0, 4.0, 9.0, 25.0], //
                [2.0, 5.0, 10.0, 26.0],
            ]]],
            &device,
        );
        let (output1, output2, output3, output4, output5) = model.forward(input.clone());

        // output1 is now scalar (f32), others are tensors
        let expected1_scalar = 26.3132618008494f32;
        let expected2 = TensorData::from([[[
            [1.0f32, 4.0, 9.0, 25.0], //
            [2.0, 5.0, 10.0, 26.0],
        ]]]);
        let expected3 = TensorData::from([[[[25.00000011f32], [26.00000011]]]]);
        let expected4 = TensorData::from([[
            [1.0f32, 4.0, 9.0, 25.0], //
            [2.0, 5.0, 10.0, 26.0],
        ]]);
        let expected5 = TensorData::from([[2.31326169f32, 5.31326169, 10.31326169, 26.31326169]]);

        // Assert scalar output
        assert_scalar_approx_eq(output1, expected1_scalar, 1e-6);
        output2
            .to_data()
            .assert_approx_eq::<FT>(&expected2, burn::tensor::Tolerance::default());
        output3
            .to_data()
            .assert_approx_eq::<FT>(&expected3, burn::tensor::Tolerance::default());
        output4
            .to_data()
            .assert_approx_eq::<FT>(&expected4, burn::tensor::Tolerance::default());
        output5
            .to_data()
            .assert_approx_eq::<FT>(&expected5, burn::tensor::Tolerance::default());
    }
}
