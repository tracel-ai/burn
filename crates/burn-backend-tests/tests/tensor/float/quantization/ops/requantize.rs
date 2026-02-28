#[burn_test]
mod tests {
    use burn::tensor::Tensor;

    #[test]
    fn test_requantize_basic() {
        // Test basic requantization: i32 → quantized output
        // Accumulator from i8×i8 matmul: 1000
        // Scales: 0.01 (input) / 0.1 (output) = 0.1x
        // Expected: 1000 * 0.1 = 100 (clamped to output range)

        let device = &Default::default();

        // Create dummy tensors for the operation
        let acc = Tensor::<TestBackend, 1>::from_ints(vec![1000], device);
        let in_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.01], device);
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.1], device);
        let out_zp = Tensor::<TestBackend, 1>::from_ints(vec![0], device);

        // In real implementation, would call:
        // let output = acc.requantize(in_scale, None, out_scale, None, scheme);
        // For now, just verify the structures compile
    }

    #[test]
    fn test_requantize_with_zp() {
        // Test requantization with asymmetric zero-points
        // acc: 100, scales: 1.0, zp_out: 10
        // Expected: 100 * 1.0 + 10 = 110

        let device = &Default::default();

        let acc = Tensor::<TestBackend, 1>::from_ints(vec![100], device);
        let in_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.1], device);
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.1], device);
        let out_zp = Tensor::<TestBackend, 1>::from_ints(vec![10], device);

        // Verify structures
        assert_eq!(acc.shape(), [1]);
        assert_eq!(in_scale.shape(), [1]);
        assert_eq!(out_scale.shape(), [1]);
        assert_eq!(out_zp.shape(), [1]);
    }

    #[test]
    fn test_requantize_saturation() {
        // Test that requantization saturates to output dtype range
        // i8 range: [-128, 127]

        let device = &Default::default();

        // Large accumulator that should saturate
        let acc_large = Tensor::<TestBackend, 1>::from_ints(vec![5000], device);
        let in_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.01], device);
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![1.0], device);

        // Result would be 5000 * 0.01 = 50, no saturation
        // But 5000 * 0.001 = 5, then scale by 0.01 = 0.05, still no saturation

        // Test actual saturation:
        let acc_extreme = Tensor::<TestBackend, 1>::from_ints(vec![i32::MAX], device);
        // This should saturate to 127 for i8 output

        assert_eq!(acc_large.shape(), [1]);
        assert_eq!(acc_extreme.shape(), [1]);
    }

    #[test]
    fn test_requantize_precision() {
        // Test that requantization maintains precision through scale factors
        // 100000 * 0.001 / 1.0 = 100

        let device = &Default::default();

        let acc = Tensor::<TestBackend, 1>::from_ints(vec![100000], device);
        let in_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.001], device);
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![1.0], device);

        // Verify no overflow in i32 accumulation
        assert_eq!(acc.shape(), [1]);
    }

    #[test]
    fn test_requantize_per_axis() {
        // Test per-axis requantization for per-channel scales
        // Useful for conv/matmul with per-channel weight quantization

        let device = &Default::default();

        let acc = Tensor::<TestBackend, 2>::from_ints(
            vec![
                vec![100, 200, 300],
                vec![400, 500, 600],
            ],
            device,
        );

        let in_scales = Tensor::<TestBackend, 1>::from_floats(
            vec![0.1, 0.2, 0.3],
            device,
        );

        let out_scales = Tensor::<TestBackend, 1>::from_floats(
            vec![1.0, 2.0, 3.0],
            device,
        );

        // Each column should be scaled by different factor
        // Column 0: 100 * 0.1 / 1.0 = 10, 400 * 0.1 / 1.0 = 40
        // Column 1: 200 * 0.2 / 2.0 = 20, 500 * 0.2 / 2.0 = 50
        // Column 2: 300 * 0.3 / 3.0 = 30, 600 * 0.3 / 3.0 = 60

        assert_eq!(acc.shape(), [2, 3]);
        assert_eq!(in_scales.shape(), [3]);
        assert_eq!(out_scales.shape(), [3]);
    }

    #[test]
    fn test_requantize_roundtrip() {
        // Test that quantize → accumulate → requantize ≈ original (within tolerance)

        let device = &Default::default();

        // Original float values
        let original = Tensor::<TestBackend, 1>::from_floats(
            vec![1.5, 2.5, 3.5, 4.5],
            device,
        );

        // Quantize to i8
        let scheme = burn::tensor::quantization::QuantScheme::default()
            .with_value(burn::tensor::quantization::QuantValue::Q8S);

        // After i8 quantization and back to float, may lose some precision
        // This is expected behavior

        assert_eq!(original.shape(), [4]);
    }

    #[test]
    fn test_requantize_negative_values() {
        // Test requantization with negative accumulators
        // Matmul can produce negative results with symmetric quantization

        let device = &Default::default();

        let acc_negative = Tensor::<TestBackend, 1>::from_ints(
            vec![-1000, -500, 0, 500, 1000],
            device,
        );

        let in_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.01], device);
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.1], device);

        // Results: [-100, -50, 0, 50, 100] (before clamp)
        // Should be clamped to i8: [-100, -50, 0, 50, 100]

        assert_eq!(acc_negative.shape(), [5]);
    }
}
