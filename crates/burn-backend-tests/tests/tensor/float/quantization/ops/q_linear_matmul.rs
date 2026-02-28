#[burn_test]
mod tests {
    use burn::tensor::{backend::Backend, Tensor, quantization::*};

    fn assert_allclose(actual: &Tensor<TestBackend, 2>, expected: &Tensor<TestBackend, 2>, tolerance: f32) {
        let diff = actual.sub(expected).abs();
        let max_diff = diff.max();
        let max_diff_val: f32 = max_diff.into_scalar();
        assert!(
            max_diff_val < tolerance,
            "Max difference {} exceeds tolerance {}",
            max_diff_val,
            tolerance
        );
    }

    #[test]
    fn test_q_linear_matmul_numeric_correctness_2x2() {
        // Validate numeric correctness: quantized matmul ≈ float matmul
        let device = &Default::default();

        // Small 2x2 matrices with known values
        let a = Tensor::<TestBackend, 2>::from_floats(
            vec![1.0, 2.0, 3.0, 4.0],
            device,
        ).reshape([2, 2]);

        let b = Tensor::<TestBackend, 2>::from_floats(
            vec![5.0, 6.0, 7.0, 8.0],
            device,
        ).reshape([2, 2]);

        // Float reference:
        // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
        // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
        let expected = a.matmul(b.clone());

        // Quantize both
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_mode(QuantMode::Symmetric);

        let a_q = a.quantize_dynamic(&scheme);
        let b_q = b.quantize_dynamic(&scheme);

        // Extract scales from quantized tensors
        let a_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.031496], device); // 4/127
        let b_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.062992], device); // 8/127
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.3937], device);  // 50/127

        // Call q_linear_matmul
        let result_q = a_q.q_linear_matmul(
            a_scale,
            None,  // No zero-point (symmetric)
            b_q,
            b_scale,
            None,  // No zero-point (symmetric)
            out_scale,
            None,  // No output zero-point
        );

        // Dequantize and compare
        let result = result_q.dequantize();

        // Allow 0.2 absolute tolerance due to quantization precision loss
        assert_allclose(&result, &expected, 0.2);
    }

    #[test]
    fn test_q_linear_matmul_identity_scales() {
        // Test with identity scales: result should match float matmul exactly (within quantization)
        let device = &Default::default();

        let a = Tensor::<TestBackend, 2>::from_floats(
            vec![2.0, 3.0, 4.0, 5.0],
            device,
        ).reshape([2, 2]);

        let b = Tensor::<TestBackend, 2>::from_floats(
            vec![1.0, 2.0, 3.0, 4.0],
            device,
        ).reshape([2, 2]);

        let expected = a.matmul(b.clone());

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_mode(QuantMode::Symmetric);

        let a_q = a.quantize_dynamic(&scheme);
        let b_q = b.quantize_dynamic(&scheme);

        // Use scales that preserve magnitude
        let a_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.0394], device);  // 5/127
        let b_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.0315], device);  // 4/127
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.1575], device); // 20/127

        let result_q = a_q.q_linear_matmul(
            a_scale,
            None,
            b_q,
            b_scale,
            None,
            out_scale,
            None,
        );

        let result = result_q.dequantize();

        // With good scale choices, should be within 0.3 absolute tolerance
        assert_allclose(&result, &expected, 0.3);
    }

    #[test]
    fn test_q_linear_matmul_vector_multiply() {
        // Validate 1x2 @ 2x1 = scalar multiplication
        let device = &Default::default();

        let a = Tensor::<TestBackend, 2>::from_floats(vec![3.0, 4.0], device).reshape([1, 2]);
        let b = Tensor::<TestBackend, 2>::from_floats(vec![5.0, 6.0], device).reshape([2, 1]);

        // Expected: 3*5 + 4*6 = 15 + 24 = 39
        let expected = a.matmul(b.clone());
        let expected_val: f32 = expected.slice([0..1, 0..1]).into_scalar();

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_mode(QuantMode::Symmetric);

        let a_q = a.quantize_dynamic(&scheme);
        let b_q = b.quantize_dynamic(&scheme);

        let a_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.0315], device);  // 4/127
        let b_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.0472], device);  // 6/127
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.3071], device); // 39/127

        let result_q = a_q.q_linear_matmul(
            a_scale,
            None,
            b_q,
            b_scale,
            None,
            out_scale,
            None,
        );

        let result = result_q.dequantize();
        let result_val: f32 = result.slice([0..1, 0..1]).into_scalar();

        // Should be very close (within 5%)
        assert!((result_val - expected_val).abs() < expected_val * 0.05,
            "Result {} != Expected {}", result_val, expected_val);
    }

    #[test]
    fn test_q_linear_matmul_different_scales_per_operand() {
        // Verify that different scales for A and B are both applied
        let device = &Default::default();

        let a = Tensor::<TestBackend, 2>::from_floats(vec![1.0, 1.0], device).reshape([1, 2]);
        let b = Tensor::<TestBackend, 2>::from_floats(vec![2.0, 2.0], device).reshape([2, 1]);

        // Expected: 1*2 + 1*2 = 4
        let expected = a.matmul(b.clone());

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_mode(QuantMode::Symmetric);

        let a_q = a.quantize_dynamic(&scheme);
        let b_q = b.quantize_dynamic(&scheme);

        // Different scales
        let a_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.0079], device);  // 1/127
        let b_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.0157], device);  // 2/127
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.0315], device); // 4/127

        let result_q = a_q.q_linear_matmul(
            a_scale,
            None,
            b_q,
            b_scale,
            None,
            out_scale,
            None,
        );

        let result = result_q.dequantize();

        assert_allclose(&result, &expected, 0.3);
    }

    #[test]
    fn test_q_linear_matmul_matrix_shape_preservation() {
        // Verify output shape is correct for different input shapes
        let device = &Default::default();

        let test_cases = vec![
            ([2, 3], [3, 4]),  // 2x3 @ 3x4 -> 2x4
            ([5, 2], [2, 6]),  // 5x2 @ 2x6 -> 5x6
            ([1, 10], [10, 1]), // 1x10 @ 10x1 -> 1x1
        ];

        for ((m, k), (_, n)) in test_cases {
            let a = Tensor::<TestBackend, 2>::ones([m, k], device);
            let b = Tensor::<TestBackend, 2>::ones([k, n], device);

            let scheme = QuantScheme::default()
                .with_value(QuantValue::Q8S)
                .with_mode(QuantMode::Symmetric);

            let a_q = a.quantize_dynamic(&scheme);
            let b_q = b.quantize_dynamic(&scheme);

            let a_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.01], device);
            let b_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.01], device);
            let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.1], device);

            let result = a_q.q_linear_matmul(
                a_scale,
                None,
                b_q,
                b_scale,
                None,
                out_scale,
                None,
            );

            let result_shape = result.dequantize().shape();
            assert_eq!(result_shape, [m, n], "Shape mismatch for {}x{} @ {}x{}", m, k, k, n);
        }
    }
}
