use core::ops::Range;

use crate as burn;
use crate::config::Config;
use crate::module::{Content, DisplaySettings, Module, ModuleDisplay};
use crate::tensor::Int;
use crate::tensor::Tensor;
use crate::tensor::backend::Backend;
use alloc::vec;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

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
        self.initialize(|x| x, device)
    }

    /// Initialize a new [RotaryEncoding](RotaryEncoding) module with a custom frequency scaling function.
    /// This is useful to apply different RoPE extensions.
    ///
    /// # Panics
    ///
    /// Panics if the size of input embedding dimension is not even.
    /// Panics if the theta parameter is not positive.
    pub fn init_with_frequency_scaling<B: Backend>(
        &self,
        scaling: impl Fn(Tensor<B, 1>) -> Tensor<B, 1>,
        device: &B::Device,
    ) -> RotaryEncoding<B> {
        self.initialize(scaling, device)
    }

    /// Initialize a new [RotaryEncoding](RotaryEncoding) module.
    ///
    /// # Panics
    ///
    /// Panics if the size of input embedding dimension is not even.
    /// Panics if the theta parameter is not positive.
    fn initialize<B: Backend>(
        &self,
        scaling: impl Fn(Tensor<B, 1>) -> Tensor<B, 1>,
        device: &B::Device,
    ) -> RotaryEncoding<B> {
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
        // `theta = 1 / (theta ^ (2i / d_model)) for i in [0..d_model/2]`
        let exponent = Tensor::<B, 1, Int>::arange_step(0..self.d_model as i64, 2, device)
            .float()
            .div_scalar(self.d_model as f32);

        // Calculate (10000 ^ (2i / d_model)) by using the log base property `exp(log(10000) * (2i / d_model))`
        // This is done since burn doesn't support exponentiation of scalar to tensor
        let theta = exponent.mul_scalar(self.theta.ln()).exp().recip();

        let theta = scaling(theta);

        let freq_complex =
            RotaryEncoding::compute_rotary_frequencies(0..self.max_sequence_length, theta.clone());

        RotaryEncoding {
            freq_complex,
            theta,
            start_offset: 0,
        }
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
#[module(custom_display)]
pub struct RotaryEncoding<B: Backend> {
    /// Complex frequency tensor of shape (max_sequence_length, d_model, 2) with real and imaginary components
    // Essentially a cache of pre-computed RoPE values.
    pub freq_complex: Tensor<B, 3>,
    /// Frequency vector used to compute/apply the complex rotations.
    pub theta: Tensor<B, 1>,
    start_offset: usize,
}

impl<B: Backend> ModuleDisplay for RotaryEncoding<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [max_sequence_length, d_model, _] = self.freq_complex.shape().dims();
        content
            .add("d_model", &d_model)
            .add("max_sequence_length", &max_sequence_length)
            .optional()
    }
}

#[allow(clippy::single_range_in_vec_init)]
impl<B: Backend> RotaryEncoding<B> {
    /// Applies rotary positional encoding to a tensor of dimensions (..., seq_len, d_model)
    ///
    /// # Arguments:
    /// * `x` - Input tensor of shape (..., seq_len, d_model). Accommodate both 3D and 4D tensors
    ///   for (batch size, seq_len, hidden_dim) or (batch size, num_heads, seq_len, hidden_dim)
    ///   respectively.
    ///
    /// # Returns:
    /// Output tensor with the same shape as input tensor after applying rotary encoding.
    ///
    /// # Panics
    /// If the input tensor does not have at least 2 dimensions for sequence length and hidden dimension.
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.apply(x, 0)
    }

    /// Applies rotary positional encoding to a tensor of dimensions (..., seq_len, d_model)
    ///
    /// # Arguments:
    /// * `x` - Input tensor of shape (..., seq_len, d_model). Accommodate both 3D and 4D tensors
    ///   for (batch size, seq_len, hidden_dim) or (batch size, num_heads, seq_len, hidden_dim)
    ///   respectively.
    /// * `start` - Sequence start position index.
    ///
    /// # Returns:
    /// Output tensor with the same shape as input tensor after applying rotary encoding.
    ///
    /// # Panics
    /// If the input tensor does not have at least 2 dimensions for sequence length and hidden dimension.
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
            Tensor::<B, 2>::from_floats([[1.0, 0.0, 0.0, 1.0], [0.0, -1.0, 1.0, 0.0]], &device);

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

    /// Shifts the pre-computed rotary frequency to cover a new range of positions.
    ///
    /// This method updates the internal frequency tensor `freq_complex` to store
    /// the rotary positional encodings for a new window of positions starting at `start`.
    pub fn shift(&mut self, start: usize) {
        let max_seq_len = self.freq_complex.dims()[0];
        assert!(
            start > self.start_offset,
            "Shift start position must be monotonically increasing"
        );

        let current_end = self.start_offset + max_seq_len;

        if start >= current_end {
            // Overwrite the whole buffer
            let new_freqs =
                Self::compute_rotary_frequencies(start..start + max_seq_len, self.theta.clone());
            self.freq_complex
                .inplace(|freqs| freqs.slice_assign([0..max_seq_len], new_freqs));
        } else {
            // Shift the tail
            let num_keep = current_end - start;
            let start_rel = start - self.start_offset;
            let tail_freqs = self.freq_complex.clone().slice([start_rel..max_seq_len]);
            self.freq_complex
                .inplace(|freqs| freqs.slice_assign([0..num_keep], tail_freqs));
            // Compute the rest and assign
            let new_freqs = Self::compute_rotary_frequencies(
                current_end..start + max_seq_len,
                self.theta.clone(),
            );
            self.freq_complex
                .inplace(|freqs| freqs.slice_assign([num_keep..max_seq_len], new_freqs));
        }
        self.start_offset = start;
    }

    /// Computes the positional rotation frequencies (cosine and sine values) used in RoPE.
    ///
    /// # Arguments
    /// - `range`: Range of position indices `[start, end)`.
    /// - `theta`: 1D tensor of shape `(d_model / 2)` containing base angular frequencies.
    ///
    /// # Returns
    /// Tensor of shape `(range.len(), d_model, 2)` containing `[cos, sin]` pairs for each position and frequency.
    fn compute_rotary_frequencies(range: Range<usize>, theta: Tensor<B, 1>) -> Tensor<B, 3> {
        let d_model = theta.dims()[0] * 2;
        let num_positions = range.end - range.start;

        // Generate frequency values for positional embeddings
        let frequencies: Tensor<B, 2> =
            Tensor::<B, 1, Int>::arange(range.start as i64..range.end as i64, &theta.device())
                .float()
                .unsqueeze()
                .transpose()
                .repeat_dim(1, d_model / 2)
                * theta.unsqueeze();

        // Convert frequency values to complex numbers (polar form)
        let p_cos = frequencies.clone().cos();
        let p_sin = frequencies.sin();

        Tensor::cat(vec![p_cos, p_sin], 1)
            .reshape([num_positions, 2, d_model / 2])
            .transpose()
            .unsqueeze_dim::<4>(2)
            .repeat_dim(2, 2)
            .reshape([num_positions, d_model, 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_rotary_encoding_forward() {
        let device = Default::default();
        let rotary_encoding = RotaryEncodingConfig::new(10, 4).init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::from_floats(
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
            .squeeze::<3>(0)
            .to_data()
            .assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
    }

    #[test]
    fn test_zero_input_rotary_encoding_forward() {
        let device = Default::default();
        let rotary_encoding = RotaryEncodingConfig::new(10, 4).init::<TestBackend>(&device);

        // Use a tensor of exact zeros as input. The output rotary embedding should be zeros as well
        let input = Tensor::<TestBackend, 4>::zeros([1, 2, 2, 4], &device);

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
            .squeeze::<3>(0)
            .to_data()
            .assert_approx_eq::<FT>(&expected_output.to_data(), Tolerance::default());
    }

    #[test]
    #[should_panic]
    fn test_valid_input_hidden_dim() {
        // Hidden dimension must be even to be able to split into real and imaginary components
        // for rotation
        let d_model = 15;
        let device = Default::default();
        let pe = RotaryEncodingConfig::new(10, d_model).init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 3>::zeros([1, 5, d_model], &device);
        let _output = pe.forward(input);
    }

    #[test]
    fn test_rotary_encoding_frequencies() {
        let device = Default::default();
        let rotary_encoding = RotaryEncodingConfig::new(2, 8).init::<TestBackend>(&device);

        let expected_freqs = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [1.0000, 0.0000],
                    [1.0000, 0.0000],
                    [1.0000, 0.0000],
                    [1.0000, 0.0000],
                ],
                [
                    [5.4030e-01, 8.4147e-01],
                    [9.9500e-01, 9.9833e-02],
                    [9.9995e-01, 9.9998e-03],
                    [9.9999e-01, 9.9999e-04],
                ],
            ],
            &device,
        )
        .unsqueeze_dim::<4>(2)
        .repeat_dim(2, 2)
        .reshape([2, 8, 2]);

        rotary_encoding
            .freq_complex
            .to_data()
            .assert_approx_eq::<FT>(&expected_freqs.to_data(), Tolerance::default());
    }

    fn apply_freq_scaling_by_parts<B: Backend>(freqs: Tensor<B, 1>) -> Tensor<B, 1> {
        // Adapted from: https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py#L45
        let scale_factor = 8.;
        let low_freq_factor = 1.;
        let high_freq_factor = 4.;
        let old_context_len = 8192.;

        let low_freq_wavelen = old_context_len / low_freq_factor;
        let high_freq_wavelen = old_context_len / high_freq_factor;

        let wavelen = freqs.clone().recip().mul_scalar(2. * core::f32::consts::PI);

        // if wavelen >= high_freq_wavelen
        let cond = wavelen.clone().greater_equal_elem(high_freq_wavelen);
        let smooth = wavelen
            .clone()
            .recip()
            .mul_scalar(old_context_len)
            .sub_scalar(low_freq_factor)
            .div_scalar(high_freq_factor - low_freq_factor);
        // (1 - smooth) * freq / scale_factor + smooth * freq
        let new_freqs = smooth
            .clone()
            .neg()
            .add_scalar(1.)
            .mul(freqs.clone().div_scalar(scale_factor))
            .add(smooth.clone().mul(freqs.clone()));
        let new_freqs = freqs.clone().mask_where(cond, new_freqs);

        // if wavelen > low_freq_wavelen
        let cond = wavelen.clone().greater_elem(low_freq_wavelen);
        let new_freqs = new_freqs.mask_where(cond, freqs.clone().div_scalar(scale_factor));

        // if wavelen < high_freq_wavelen
        let cond = wavelen.lower_elem(high_freq_wavelen);
        new_freqs.mask_where(cond, freqs)
    }

    #[test]
    fn test_rotary_encoding_with_frequency_scaling() {
        let device = Default::default();
        let rotary_encoding = RotaryEncodingConfig::new(2, 8)
            .init_with_frequency_scaling::<TestBackend>(apply_freq_scaling_by_parts, &device);

        let expected_freqs = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [1.0000, 0.0000],
                    [1.0000, 0.0000],
                    [1.0000, 0.0000],
                    [1.0000, 0.0000],
                ],
                [
                    [5.4030e-01, 8.4148e-01],
                    [9.9500e-01, 9.9833e-02],
                    [9.9995e-01, 9.9998e-03],
                    [1.0000, 2.1361e-04],
                ],
            ],
            &device,
        )
        .unsqueeze_dim::<4>(2)
        .repeat_dim(2, 2)
        .reshape([2, 8, 2]);

        rotary_encoding
            .freq_complex
            .to_data()
            .assert_approx_eq::<FT>(&expected_freqs.to_data(), Tolerance::default());
    }

    #[test]
    fn test_rotary_encoding_shift_full() {
        let device = Default::default();
        let rotary_encoding = RotaryEncodingConfig::new(10, 4).init::<TestBackend>(&device);

        // Input = [Batch size, Num of heads, Seq_len, d_model]
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
            ],
            &device,
        )
        .unsqueeze::<4>();

        // Initializing for a bigger cache (e.g., max_seq_len = 10) should give the same result
        // as using a smaller cache of pre-computed RoPE frequencies that are shifted to the same
        // initial position
        let expected_output = rotary_encoding.apply(input.clone(), 6);

        let mut rotary_encoding = RotaryEncodingConfig::new(4, 4).init::<TestBackend>(&device);
        rotary_encoding.shift(6); // start > 4 will perform a full re-compute

        let output = rotary_encoding.apply(input, 0);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected_output.into_data(), Tolerance::default());
    }

    #[test]
    fn test_rotary_encoding_shift() {
        let device = Default::default();
        let rotary_encoding = RotaryEncodingConfig::new(10, 4).init::<TestBackend>(&device);

        // Input = [Batch size, Num of heads, Seq_len, d_model]
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
            ],
            &device,
        )
        .unsqueeze::<4>();

        // Initializing for a bigger cache (e.g., max_seq_len = 10) should give the same result
        // as using a smaller cache of pre-computed RoPE frequencies that are shifted to the same
        // initial position
        let expected_output = rotary_encoding.apply(input.clone(), 2);

        let mut rotary_encoding = RotaryEncodingConfig::new(4, 4).init::<TestBackend>(&device);
        rotary_encoding.shift(2); // start < 4 will shift the (current_end - start) freqs and compute the rest

        let output = rotary_encoding.apply(input, 0);

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected_output.into_data(), Tolerance::default());
    }

    #[test]
    fn test_rotary_encoding_shift_multiple() {
        let device = Default::default();
        let mut rotary_encoding = RotaryEncodingConfig::new(4, 4).init::<TestBackend>(&device);
        rotary_encoding.shift(2);
        rotary_encoding.shift(5);
    }

    #[test]
    #[should_panic = "Shift start position must be monotonically increasing"]
    fn test_rotary_encoding_shift_should_increase() {
        let device = Default::default();
        let mut rotary_encoding = RotaryEncodingConfig::new(4, 4).init::<TestBackend>(&device);
        rotary_encoding.shift(6);
        rotary_encoding.shift(4); // should be monotonically increasing
    }

    #[test]
    fn display() {
        let config = RotaryEncodingConfig::new(10, 4);
        let pe = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{pe}"),
            "RotaryEncoding {d_model: 4, max_sequence_length: 10}"
        );
    }
}
