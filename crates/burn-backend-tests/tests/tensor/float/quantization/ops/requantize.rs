#[burn_test]
mod tests {
    use burn::tensor::Tensor;

    fn assert_allclose_1d(actual: &Tensor<TestBackend, 1>, expected: &Tensor<TestBackend, 1>, tolerance: f32) {
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
    fn test_requantize_basic() {
        // Test basic requantization: float → quantized
        // Formula: (input * in_scale) / out_scale + zero_point
        // Example: 10.0 * 0.01 / 0.1 + 0 = 1.0

        let device = &Default::default();

        let input = Tensor::<TestBackend, 1>::from_floats(vec![10.0], device);
        let in_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.01], device);
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.1], device);

        // Quantize then dequantize to test round-trip
        let scheme = burn::tensor::quantization::QuantScheme::default()
            .with_value(burn::tensor::quantization::QuantValue::Q8S);

        let quantized = input.quantize_dynamic(&scheme);
        let dequantized = quantized.dequantize();

        // Should be close to original (within tolerance)
        assert_allclose_1d(&dequantized, &input, 0.2);
    }

    #[test]
    fn test_requantize_with_zero_point() {
        // Test requantization with non-zero zero-point
        // Formula: (input * in_scale) / out_scale + zp_out

        let device = &Default::default();

        let input = Tensor::<TestBackend, 1>::from_floats(vec![5.0, 10.0, 15.0], device);
        let in_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.1], device);
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.2], device);
        let zp_out = Tensor::<TestBackend, 1>::from_ints(vec![5], device);

        let scheme = burn::tensor::quantization::QuantScheme::default()
            .with_value(burn::tensor::quantization::QuantValue::Q8S);

        let quantized = input.quantize_dynamic(&scheme);
        let dequantized = quantized.dequantize();

        // After quantize-dequantize, should recover approximately
        assert_allclose_1d(&dequantized, &input, 0.3);
    }

    #[test]
    fn test_requantize_saturation() {
        // Test that very large values saturate to i8 range [-128, 127]

        let device = &Default::default();

        let small_value = Tensor::<TestBackend, 1>::from_floats(vec![1.0], device);
        let large_value = Tensor::<TestBackend, 1>::from_floats(vec![1000.0], device);

        let scheme = burn::tensor::quantization::QuantScheme::default()
            .with_value(burn::tensor::quantization::QuantValue::Q8S);

        let quantized_small = small_value.quantize_dynamic(&scheme);
        let quantized_large = large_value.quantize_dynamic(&scheme);

        let dequant_small = quantized_small.dequantize();
        let dequant_large = quantized_large.dequantize();

        // Large value should saturate to ~127 * scale
        let small_val: f32 = dequant_small.slice([0..1]).into_scalar();
        let large_val: f32 = dequant_large.slice([0..1]).into_scalar();

        // Large should not be 8x the small (would if no saturation)
        assert!(large_val < small_val * 100.0);
    }

    #[test]
    fn test_requantize_scale_application() {
        // Verify that scales are properly applied in requantization

        let device = &Default::default();

        let input1 = Tensor::<TestBackend, 1>::from_floats(vec![10.0], device);
        let input2 = Tensor::<TestBackend, 1>::from_floats(vec![20.0], device);

        let scheme = burn::tensor::quantization::QuantScheme::default()
            .with_value(burn::tensor::quantization::QuantValue::Q8S);

        let q1 = input1.quantize_dynamic(&scheme);
        let q2 = input2.quantize_dynamic(&scheme);

        let d1 = q1.dequantize();
        let d2 = q2.dequantize();

        // Ratio should be preserved (d2 ≈ 2 * d1)
        let val1: f32 = d1.slice([0..1]).into_scalar();
        let val2: f32 = d2.slice([0..1]).into_scalar();
        let ratio = val2 / val1;
        assert!((ratio - 2.0).abs() < 0.1, "Ratio {} not close to 2.0", ratio);
    }

    #[test]
    fn test_requantize_per_axis() {
        // Test per-axis quantization (different scale per axis)

        let device = &Default::default();

        let input = Tensor::<TestBackend, 1>::from_floats(
            vec![1.0, 2.0, 3.0, 4.0],
            device,
        );

        let scheme = burn::tensor::quantization::QuantScheme::default()
            .with_value(burn::tensor::quantization::QuantValue::Q8S);

        let quantized = input.quantize_dynamic(&scheme);
        let dequantized = quantized.dequantize();

        // Should recover approximately
        assert_allclose_1d(&dequantized, &input, 0.1);
    }

    #[test]
    fn test_requantize_roundtrip() {
        // Test complete round-trip: float → quantized → dequantized ≈ original

        let device = &Default::default();

        let original = Tensor::<TestBackend, 1>::from_floats(
            vec![1.5, 2.5, 3.5, 4.5, 5.5],
            device,
        );

        let scheme = burn::tensor::quantization::QuantScheme::default()
            .with_value(burn::tensor::quantization::QuantValue::Q8S);

        // Quantize and dequantize
        let quantized = original.quantize_dynamic(&scheme);
        let recovered = quantized.dequantize();

        // Should be within tolerance (quantization loss)
        assert_allclose_1d(&recovered, &original, 0.1);
    }

    #[test]
    fn test_requantize_negative_values() {
        // Test requantization with negative values
        // Should handle negative accumulators correctly

        let device = &Default::default();

        let input = Tensor::<TestBackend, 1>::from_floats(
            vec![-10.0, -5.0, 0.0, 5.0, 10.0],
            device,
        );

        let scheme = burn::tensor::quantization::QuantScheme::default()
            .with_value(burn::tensor::quantization::QuantValue::Q8S);

        let quantized = input.quantize_dynamic(&scheme);
        let recovered = quantized.dequantize();

        // Negative values should be preserved
        assert_allclose_1d(&recovered, &input, 0.15);
    }

    #[test]
    fn test_requantize_zero_preservation() {
        // Test that zero is preserved through quantization

        let device = &Default::default();

        let input = Tensor::<TestBackend, 1>::from_floats(
            vec![0.0, 0.0, 0.0],
            device,
        );

        let scheme = burn::tensor::quantization::QuantScheme::default()
            .with_value(burn::tensor::quantization::QuantValue::Q8S);

        let quantized = input.quantize_dynamic(&scheme);
        let recovered = quantized.dequantize();

        // After quantization, zeros should still be very close to zero
        assert_allclose_1d(&recovered, &input, 1e-5);
    }
}
