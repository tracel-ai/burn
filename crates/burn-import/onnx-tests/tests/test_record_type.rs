// This test suite verifies that the exported models are compatible with the
// different record types. It uses an existing model (conv1d.onnx) and exports
// it with different record types. Then it loads the exported model and runs it
// with the same input to verify that the output is the same.
// For half precision, we use a different tolerance because the output is
// different.

macro_rules! test_model {
    ($mod_name:ident) => {
        test_model!($mod_name, 1.0e-4); // Default tolerance
    };
    ($mod_name:ident, $tolerance:expr) => {
        pub mod $mod_name {
            include!(concat!(
                env!("OUT_DIR"),
                "/model/",
                stringify!($mod_name),
                "/conv1d.rs"
            ));
        }

        #[test]
        fn $mod_name() {
            // Initialize the model with weights (loaded from the exported file)
            let model: $mod_name::Model<Backend> = $mod_name::Model::default();

            // Run the model with pi as input for easier testing
            let input = Tensor::<Backend, 3>::full([6, 4, 10], consts::PI, &Default::default());

            let output = model.forward(input);

            // test the output shape
            let expected_shape = Shape::from([6, 2, 7]);
            assert_eq!(output.shape(), expected_shape);

            // We are using the sum of the output tensor to test the correctness of the conv1d node
            // because the output tensor is too large to compare with the expected tensor.
            let output_sum = output.sum().into_scalar();
            let expected_sum = -54.549_243; // from pytorch
            assert!(expected_sum.approx_eq(output_sum, ($tolerance, 2)));
        }
    };
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Shape, Tensor};
    use float_cmp::ApproxEq;
    use std::f64::consts;

    type Backend = burn_ndarray::NdArray<f32>;

    test_model!(named_mpk);
    test_model!(named_mpk_half, 1.0e-2); // Reduce tolerance for half precision
    test_model!(pretty_json);
    test_model!(pretty_json_half, 1.0e-2); // Reduce tolerance for half precision
    test_model!(named_mpk_gz);
    test_model!(named_mpk_gz_half, 1.0e-2); // Reduce tolerance for half precision
    test_model!(bincode);
    test_model!(bincode_half, 1.0e-2); // Reduce tolerance for half precision
    test_model!(bincode_embedded);
    test_model!(bincode_embedded_half, 1.0e-2); // Reduce tolerance for half precision
}
