use crate as burn;
use crate::config::Config;
use crate::module::Module;
use crate::tensor::backend::Backend;
use crate::tensor::Int;
use crate::tensor::Tensor;
use alloc::vec;

#[cfg(not(feature = "std"))]
use num_traits::Float;

/// Configuration to create a [RotaryEncoding](RotaryEncoding) layer using the [init function](RotaryEncodingConfig::init).
#[derive(Config, Debug)]
pub struct RotaryEncodingConfig {
    /// Maximum sequence length of input
    pub max_sequence_length: usize,

    /// Size of the input embedding or hidden dimension
    pub d_model: usize,

    /// Scaling factor for frequency computation. Defaults to 10000.0
    #[config(default = "10000.0")]
    pub theta: f32,
}

impl RotaryEncodingConfig {
    /// Initialize a new [RotaryEncoding](RotaryEncoding) module.
    ///
    /// # Panics
    ///
    /// Panics if the size of input embedding dimension is not even.
    /// Panics if the theta parameter is not positive.
    pub fn init<B: Backend>(&self, device: &B::Device) -> RotaryEncoding<B> {
        assert_eq!(
            self.d_model % 2,
            0,
            "The input embedding dimension must be even"
        );
        assert!(
            self.theta > 0.0,
            "Theta parameter must be positive (default: 10000)."
        );

        // Calculate the rotation frequencies for positional embeddings based on the formula
        // `theta_i = 1 / (10000 ^ (2i / d_model)) for i in [0..d_model/2]`
        let exponent = Tensor::<B, 1, Int>::arange_step(0..self.d_model as i64, 2, device)
            .float()
            .div_scalar(self.d_model as f32);

        // Calculate (10000 ^ (2i / d_model)) by using the log base property `exp(log(10000) * (2i / d_model))`
        // This is done since burn doesn't support exponentiation of scalar to tensor
        let theta_i = exponent.mul_scalar(self.theta.ln()).exp();
        let theta_i = theta_i.powf_scalar(-1.0);

        // Generate frequency values for positional embeddings
        let frequencies: Tensor<B, 2> =
            Tensor::<B, 1, Int>::arange(0..self.max_sequence_length as i64, device)
                .float()
                .unsqueeze()
                .transpose()
                .repeat(1, self.d_model / 2)
                * theta_i.unsqueeze();

        // Convert frequency values to complex numbers (polar form)
        let p_cos = frequencies.clone().cos();
        let p_sin = frequencies.sin();

        // Create the frequency tensor of shape (max_sequence_length, d_model, 2) with the real(cos)
        // and imaginary(sin) components along last dimension
        let freq_complex: Tensor<B, 3> = Tensor::cat(vec![p_cos, p_sin], 1)
            .reshape([self.max_sequence_length, 2, self.d_model / 2])
            .transpose()
            .unsqueeze_dim::<4>(2)
            .repeat(2, 2)
            .reshape([self.max_sequence_length, self.d_model, 2]);

        RotaryEncoding { freq_complex }
    }
}

/// A module that applies rotary positional encoding to a tensor.
/// Rotary Position Encoding or Embedding (RoPE), is a type of position embedding which encodes
/// absolute positional information with rotation matrix and naturally incorporates
/// explicit relative position dependency in self-attention formulation.
///
/// Introduced in the paper: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
///
/// Should be created using [RotaryEncodingConfig].
#[derive(Module, Debug)]
pub struct RotaryEncoding<B: Backend> {
    /// Frequency Tensor of shape (max_sequence_length, d_model, 2) with real and imaginary components
    freq_complex: Tensor<B, 3>,
}

#[allow(clippy::single_range_in_vec_init)]
impl<B: Backend> RotaryEncoding<B> {
    /// Applies rotary positional encoding to a tensor of dimensions (..., seq_len, d_model)
    ///
    /// Arguments:
    /// * `x` - Input tensor of shape (..., seq_len, d_model). Accommodate both 3D and 4D tensors
    /// for (batch size, seq_len, hidden_dim) or (batch size, num_heads, seq_len, hidden_dim)
    /// respectively.
    ///
    /// Returns:
    /// * Output tensor with the same shape as input tensor after applying rotary encoding.
    ///
    /// Panics if the input tensor does not have at least 2 dimensions for sequence length and hidden dimension.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.apply(x, 0)
    }

    /// Applies rotary positional encoding to a tensor of dimensions (..., seq_len, d_model)
    ///
    /// Arguments:
    /// * `x` - Input tensor of shape (..., seq_len, d_model). Accommodate both 3D and 4D tensors
    /// for (batch size, seq_len, hidden_dim) or (batch size, num_heads, seq_len, hidden_dim)
    /// respectively.
    /// * `start` - Sequence start position index.
    ///
    /// Returns:
    /// * Output tensor with the same shape as input tensor after applying rotary encoding.
    ///
    /// Panics if the input tensor does not have at least 2 dimensions for sequence length and hidden dimension.
    pub fn apply<const D: usize>(&self, x: Tensor<B, D>, start: usize) -> Tensor<B, D> {
        assert!(
            D >= 2,
            "Input tensor must have at least 2 dimensions for sequence length and hidden dimension"
        );

        let device = x.device();
        let input_shape = x.shape();

        // Extract the sequence length and embedding dimension, other dimensions are kept generic
        // to allow both 3D and 4D tensors i.e. batch_size or (batch_size, num_heads)
        let (seq_len, d_model) = (x.dims()[D - 2], x.dims()[D - 1]);
        let dummy_dim_size = input_shape.num_elements() / (seq_len * d_model);

        // Create a dummy tensor with signed ones based on the 2D rotation matrix
        // [[cos, -sin], [sin, cos]]
        let sign_tensor =
            Tensor::from_floats([[1.0, 0.0, 0.0, 1.0], [0.0, -1.0, 1.0, 0.0]], &device);

        // Rotate input using the frequency tensor. Slice the frequencies till input sequence length
        let out: Tensor<B, 4> = x
            .reshape([dummy_dim_size, seq_len, d_model / 2, 2])
            .matmul(sign_tensor.unsqueeze())
            .reshape([dummy_dim_size, seq_len, d_model, 2])
            * self
                .freq_complex
                .clone()
                .slice([start..start + seq_len])
                .unsqueeze();

        // Sum the real and imaginary components to get output tensor and reshape to original shape
        out.sum_dim(D - 1).reshape(input_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_rotary_encoding_forward() {
        let device = Default::default();
        let rotary_encoding = RotaryEncodingConfig::new(10, 4).init::<TestBackend>(&device);

        let input = Tensor::from_floats(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
            ],
            &device,
        );

        // Input = [Batch size, Num of heads, Seq_len, d_model]
        let input = input.unsqueeze::<4>();

        let output = rotary_encoding.forward(input);
        let expected_output = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [1.0000, 2.0000, 3.0000, 4.0000],
                    [-2.3473, 7.4492, 6.9197, 8.0696],
                ],
                [
                    [9.0000, 10.0000, 11.0000, 12.0000],
                    [-4.7567, 18.5034, 14.8393, 16.1492],
                ],
            ],
            &device,
        );

        output
            .squeeze(0)
            .to_data()
            .assert_approx_eq(&expected_output.to_data(), 4);
    }

    #[test]
    fn test_zero_input_rotary_encoding_forward() {
        let device = Default::default();
        let rotary_encoding = RotaryEncodingConfig::new(10, 4).init::<TestBackend>(&device);

        // Use a tensor of exact zeros as input. The output rotary embedding should be zeros as well
        let input = Tensor::zeros([1, 2, 2, 4], &device);

        let output = rotary_encoding.forward(input);
        let expected_output = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000],
                ],
                [
                    [0.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000],
                ],
            ],
            &device,
        );

        output
            .squeeze(0)
            .to_data()
            .assert_approx_eq(&expected_output.to_data(), 4);
    }

    #[test]
    #[should_panic]
    fn test_valid_input_hidden_dim() {
        // Hidden dimension must be even to be able to split into real and imaginary components
        // for rotation
        let d_model = 15;
        let device = Default::default();
        let pe = RotaryEncodingConfig::new(10, d_model).init::<TestBackend>(&device);
        let input = Tensor::zeros([1, 5, d_model], &device);
        let _output = pe.forward(input);
    }
}
