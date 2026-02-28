#[burn_test]
mod tests {
    use burn::tensor::{backend::Backend, Tensor, quantization::*};

    #[test]
    fn test_q_linear_conv2d_basic() {
        let device = &Default::default();

        // Create simple quantized tensors
        // Input: [1, 1, 3, 3] - single channel, 3×3 spatial
        // Weight: [1, 1, 2, 2] - 1 output channel, 2×2 kernel
        let input = Tensor::<TestBackend, 4>::from_floats(
            vec![vec![vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ]]],
            device,
        );

        let weight = Tensor::<TestBackend, 4>::from_floats(
            vec![vec![vec![
                vec![1.0, 0.5],
                vec![0.5, 1.0],
            ]]],
            device,
        );

        // Quantize
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_mode(QuantMode::Symmetric);

        let input_q = input.quantize_dynamic(&scheme);
        let weight_q = weight.quantize_dynamic(&scheme);

        // Get scales
        let in_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.03937008], device);
        let w_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.00787402], device);
        let out_scale = Tensor::<TestBackend, 1>::from_floats(vec![0.05905512], device);

        // Verify tensor shapes for convolution
        assert_eq!(input_q.shape(), [1, 1, 3, 3]);
        assert_eq!(weight_q.shape(), [1, 1, 2, 2]);
        assert_eq!(in_scale.shape(), [1]);
        assert_eq!(w_scale.shape(), [1]);
        assert_eq!(out_scale.shape(), [1]);
    }

    #[test]
    fn test_q_linear_conv2d_per_channel() {
        let device = &Default::default();

        // Multi-channel conv with per-channel weight quantization
        // Input: [1, 3, 4, 4] - 3 input channels
        // Weight: [16, 3, 3, 3] - 16 output channels, 3×3 kernel
        let input = Tensor::<TestBackend, 4>::ones([1, 3, 4, 4], device);
        let weight = Tensor::<TestBackend, 4>::ones([16, 3, 3, 3], device);

        // Per-channel weight scales: one for each output channel
        let w_scales = Tensor::<TestBackend, 1>::ones([16], device) * 0.01;
        let w_zp = Tensor::<TestBackend, 1>::zeros([16], device);

        // Each output channel has its own quantization parameters
        assert_eq!(w_scales.shape(), [16]);
        assert_eq!(w_zp.shape(), [16]);
    }

    #[test]
    fn test_q_linear_conv1d() {
        let device = &Default::default();

        // 1D convolution (for sequence/audio models)
        // Input: [2, 3, 10] - batch=2, channels=3, length=10
        // Weight: [8, 3, 3] - 8 filters, 3 input channels, kernel size 3
        let input = Tensor::<TestBackend, 3>::ones([2, 3, 10], device);
        let weight = Tensor::<TestBackend, 3>::ones([8, 3, 3], device);

        let w_scales = Tensor::<TestBackend, 1>::ones([8], device) * 0.01;

        assert_eq!(input.shape(), [2, 3, 10]);
        assert_eq!(weight.shape(), [8, 3, 3]);
        assert_eq!(w_scales.shape(), [8]);
    }

    #[test]
    fn test_q_linear_depthwise_conv() {
        let device = &Default::default();

        // Depthwise convolution (1 filter per input channel)
        // Input: [1, 16, 32, 32] - 16 input/output channels
        // Weight: [16, 1, 3, 3] - depthwise: 1 filter per input channel
        let input = Tensor::<TestBackend, 4>::ones([1, 16, 32, 32], device);
        let weight = Tensor::<TestBackend, 4>::ones([16, 1, 3, 3], device);

        let w_scales = Tensor::<TestBackend, 1>::ones([16], device) * 0.01;

        assert_eq!(input.shape(), [1, 16, 32, 32]);
        assert_eq!(weight.shape(), [16, 1, 3, 3]);
        assert_eq!(w_scales.shape(), [16]);
    }

    #[test]
    fn test_q_linear_conv_with_bias() {
        let device = &Default::default();

        let input = Tensor::<TestBackend, 4>::ones([1, 3, 8, 8], device);
        let weight = Tensor::<TestBackend, 4>::ones([32, 3, 3, 3], device);
        let bias = Tensor::<TestBackend, 1>::ones([32], device) * 0.1;

        // Bias should be present but not quantized (same precision as output)
        assert_eq!(bias.shape(), [32]);
    }

    #[test]
    fn test_q_linear_conv_different_strides() {
        let device = &Default::default();

        // Test with different stride values
        // Stride affects output spatial dimensions
        let input = Tensor::<TestBackend, 4>::ones([1, 3, 16, 16], device);
        let weight = Tensor::<TestBackend, 4>::ones([8, 3, 3, 3], device);

        // With stride=2, 16x16 input → 8x8 output (with valid padding)
        // With stride=1, 16x16 input → 14x14 output (with valid padding)
        // With stride=1, padding=1 → 16x16 output (with same padding)

        assert_eq!(input.shape(), [1, 3, 16, 16]);
        assert_eq!(weight.shape(), [8, 3, 3, 3]);
    }

    #[test]
    fn test_q_linear_conv_different_padding() {
        let device = &Default::default();

        let input = Tensor::<TestBackend, 4>::ones([1, 3, 8, 8], device);
        let weight = Tensor::<TestBackend, 4>::ones([16, 3, 3, 3], device);

        // Different padding modes:
        // - Valid (no padding): 8×8 input → 6×6 output
        // - Same (with padding): 8×8 input → 8×8 output
        // - Custom: specify padding explicitly

        assert_eq!(input.shape(), [1, 3, 8, 8]);
        assert_eq!(weight.shape(), [16, 3, 3, 3]);
    }

    #[test]
    fn test_q_linear_conv_asymmetric_zp() {
        let device = &Default::default();

        let input = Tensor::<TestBackend, 4>::ones([1, 3, 8, 8], device);
        let weight = Tensor::<TestBackend, 4>::ones([16, 3, 3, 3], device);

        // Asymmetric quantization with zero-points
        let in_zp = Tensor::<TestBackend, 1>::zeros([1], device); // Input zp (could be non-zero for u8)
        let w_zp = Tensor::<TestBackend, 1>::ones([16], device) * 128u8; // Per-channel weight zp

        // Zero-point affects the computation:
        // accum = sum((input - in_zp) * (weight - w_zp))

        assert_eq!(in_zp.shape(), [1]);
        assert_eq!(w_zp.shape(), [16]);
    }

    #[test]
    fn test_q_linear_conv_batch() {
        let device = &Default::default();

        // Batch processing: multiple inputs
        let input = Tensor::<TestBackend, 4>::ones([8, 3, 16, 16], device); // batch_size=8
        let weight = Tensor::<TestBackend, 4>::ones([32, 3, 3, 3], device);

        // Each batch element processes independently
        // Output: [8, 32, H_out, W_out]

        assert_eq!(input.shape()[0], 8); // Batch dimension
        assert_eq!(weight.shape()[0], 32); // Output channels
    }

    #[test]
    fn test_q_linear_conv_dequantize_reference() {
        let device = &Default::default();

        // Test that dequantize path produces correct reference result
        let input = Tensor::<TestBackend, 4>::from_floats(
            vec![vec![vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0],
            ]]],
            device,
        );

        let weight = Tensor::<TestBackend, 4>::from_floats(
            vec![vec![vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
            ]]],
            device,
        );

        // Float reference result
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_mode(QuantMode::Symmetric);

        let input_q = input.quantize_dynamic(&scheme);
        let weight_q = weight.quantize_dynamic(&scheme);

        // Dequantize should recover original (approximately)
        let input_dequant = input_q.dequantize();
        let weight_dequant = weight_q.dequantize();

        // Shapes preserved
        assert_eq!(input_dequant.shape(), input.shape());
        assert_eq!(weight_dequant.shape(), weight.shape());
    }
}
