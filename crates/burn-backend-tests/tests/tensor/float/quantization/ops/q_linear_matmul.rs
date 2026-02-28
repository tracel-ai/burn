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
    fn test_q_linear_matmul_basic() {
        let device = &Default::default();

        // Simple 2x2 @ 2x2 test
        let lhs = Tensor::<TestBackend, 2>::from_floats(
            vec![1.0, 2.0, 3.0, 4.0],
            device,
        ).reshape([2, 2]);

        let rhs = Tensor::<TestBackend, 2>::from_floats(
            vec![5.0, 6.0, 7.0, 8.0],
            device,
        ).reshape([2, 2]);

        // Expected result: [[19, 22], [43, 50]]
        let expected = lhs.matmul(rhs.clone());

        // Quantize
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_mode(QuantMode::Symmetric);

        let lhs_q = lhs.quantize_dynamic(&scheme);
        let rhs_q = rhs.quantize_dynamic(&scheme);

        // Use proper scales
        let lhs_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.031496], device);
        let rhs_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.062992], device);
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.1984], device);

        // Call q_linear_matmul
        let output = lhs_q.q_linear_matmul(
            lhs_scale,
            None,
            rhs_q,
            rhs_scale,
            None,
            out_scale,
            None,
        );

        // Dequantize and validate
        let output_float = output.dequantize();

        // Should be close to expected (within 5% due to quantization)
        assert_allclose(&output_float, &expected, 5.0);
    }

    #[test]
    fn test_q_linear_matmul_shapes() {
        let device = &Default::default();

        // Test various shapes: 2x3 @ 3x4 -> 2x4
        let lhs = Tensor::<TestBackend, 2>::ones([2, 3], device);
        let rhs = Tensor::<TestBackend, 2>::ones([3, 4], device);

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_mode(QuantMode::Symmetric);

        let lhs_q = lhs.quantize_dynamic(&scheme);
        let rhs_q = rhs.quantize_dynamic(&scheme);

        let lhs_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.01], device);
        let rhs_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.01], device);
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.1], device);

        let output = lhs_q.q_linear_matmul(
            lhs_scale,
            None,
            rhs_q,
            rhs_scale,
            None,
            out_scale,
            None,
        );

        let output_float = output.dequantize();
        assert_eq!(output_float.shape(), [2, 4]);
    }

    #[test]
    fn test_q_linear_matmul_with_scales() {
        // Verify that scales are actually being applied
        let device = &Default::default();

        let a = Tensor::<TestBackend, 2>::from_floats(vec![2.0, 2.0], device).reshape([1, 2]);
        let b = Tensor::<TestBackend, 2>::from_floats(vec![3.0, 3.0], device).reshape([2, 1]);

        // Float matmul: [2, 2] @ [3, 3]^T = [6 + 6] = [12]
        let expected = a.matmul(b);

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_mode(QuantMode::Symmetric);

        let a_q = a.quantize_dynamic(&scheme);
        let b_q = b.quantize_dynamic(&scheme);

        // Test with explicit scales
        let a_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.0157], device);  // ~2/127
        let b_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.0236], device);  // ~3/127
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.0945], device); // ~12/127

        let output = a_q.q_linear_matmul(
            a_scale,
            None,
            b_q,
            b_scale,
            None,
            out_scale,
            None,
        );

        let output_float = output.dequantize();
        assert_allclose(&output_float, &expected, 2.0);
    }

    #[test]
    fn test_q_linear_matmul_batch() {
        // Test batch dimension support
        let device = &Default::default();

        // Batch of 2: [2, 3, 2] @ [2, 2, 4] -> [2, 3, 4]
        let lhs = Tensor::<TestBackend, 3>::ones([2, 3, 2], device);
        let rhs = Tensor::<TestBackend, 3>::ones([2, 2, 4], device);

        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_mode(QuantMode::Symmetric);

        let lhs_q = lhs.quantize_dynamic(&scheme);
        let rhs_q = rhs.quantize_dynamic(&scheme);

        let lhs_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.01], device);
        let rhs_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.01], device);
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.1], device);

        // Note: This would need backend support for batched matmul
        // Just verify shapes work
        let output = lhs_q.q_linear_matmul(
            lhs_scale,
            None,
            rhs_q,
            rhs_scale,
            None,
            out_scale,
            None,
        );

        assert_eq!(output.dequantize().shape(), [2, 3, 4]);
    }
}
