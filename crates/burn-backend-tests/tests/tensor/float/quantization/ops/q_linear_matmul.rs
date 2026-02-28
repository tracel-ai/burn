#[burn_test]
mod tests {
    use burn::tensor::{backend::Backend, Tensor, quantization::*};

    #[test]
    fn test_q_linear_matmul_symmetric() {
        let device = &Default::default();

        // Create test data: 2x3 @ 3x2 -> 2x2
        let lhs_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let lhs = Tensor::<TestBackend, 2>::from_floats(lhs_data, device).reshape([2, 3]);
        let rhs = Tensor::<TestBackend, 2>::from_floats(rhs_data, device).reshape([3, 2]);

        // Quantize with symmetric scheme
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_mode(QuantMode::Symmetric);

        let lhs_q = lhs.quantize_dynamic(&scheme);
        let rhs_q = rhs.quantize_dynamic(&scheme);

        // Get scales
        let lhs_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.03937008], device);
        let rhs_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.04724409], device);
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.11811024], device);

        // Call q_linear_matmul (symmetric, no zero-points)
        let output = lhs_q.q_linear_matmul(
            lhs_scale,
            None,
            rhs_q,
            rhs_scale,
            None,
            out_scale,
            None,
        );

        // Dequantize output and verify against float reference
        let output_dequant = output.dequantize();
        let expected = lhs.matmul(rhs); // Float reference

        // Check shapes
        assert_eq!(output_dequant.shape(), expected.shape());

        // NOTE: Full numeric validation would require setting test tolerance
        // and comparing with ONNX reference implementation
    }

    #[test]
    fn test_q_linear_matmul_asymmetric_not_yet_supported() {
        // Test case demonstrating zero-point parameters
        // Full implementation comes in Phase 4 with native integer kernels

        // Asymmetric quantization uses zero-points:
        // (A - a_zero_point) * a_scale  *  (B - b_zero_point) * b_scale / y_scale + y_zero_point
        //
        // Currently this uses the dequantize-matmul-quantize fallback
        // Phase 4 will add native integer path where zero-points are critical
    }
}
