use alloc::vec::Vec;

use crate as burn;
use crate::config::Config;
use crate::module::Module;
use crate::tensor::backend::Backend;
use crate::tensor::Data;
use crate::tensor::Tensor;

#[cfg(not(feature = "std"))]
use num_traits::Float;

/// Configuration to create a [PositionalEncoding](PositionalEncoding) layer using the [init function](PositionalEncodingConfig::init).
#[derive(Config)]
pub struct PositionalEncodingConfig {
    /// Maximum sequence size to use.
    #[config(default = "5_000")]
    pub max_sequence_size: usize,

    /// The size of each vector.
    pub d_model: usize,

    /// Max time scale to use.
    #[config(default = "10_000")]
    pub max_timescale: usize,
}

/// Positional encoding layer for transformer models.
///
/// This layer adds positional information to the input embeddings, allowing the transformer model
/// to take into account the order of the sequence. The positional encoding is added to the input
/// embeddings by computing a set of sinusoidal functions with different frequencies and phases.
///
/// Sinusoids are used for positional embedding introduced in
/// [Attention is all you need](https://arxiv.org/abs/1706.03762).
///
/// The reference implementation can be found here:
/// [LANGUAGE MODELING WITH NN.TRANSFORMER AND TORCHTEXT
/// ](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
///
/// Should be created using [PositionalEncodingConfig]
#[derive(Module, Debug)]
pub struct PositionalEncoding<B: Backend> {
    sinusoids: Tensor<B, 3>,
}

impl PositionalEncodingConfig {
    /// Initialize a new [PositionalEncoding](PositionalEncoding) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PositionalEncoding<B> {
        let sinusoids = generate_sinusoids::<B>(
            self.max_sequence_size,
            self.d_model,
            self.max_timescale,
            device,
        )
        .unsqueeze::<3>();

        PositionalEncoding { sinusoids }
    }
}

impl<B: Backend> PositionalEncoding<B> {
    /// Applies the forward pass on the input tensor by adding the sinusoids to the input.
    ///
    /// # Shapes
    ///
    /// * input: [batch_size, seq_length, d_model]
    /// * output: [batch_size, seq_length, d_model]
    ///
    ///
    /// # Panics
    ///
    /// * Panics if the input sequence length is greater than the maximum sequence size.
    /// * Panics if the input d_model is not equal to the d_model of the sinusoids.
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, seq_length, d_model_input] = input.dims();

        let [batch_size, max_sequence_size, d_model] = self.sinusoids.dims();

        assert!(
            max_sequence_size >= seq_length,
            "max_sequence_size({}) must be greater or equal than length({seq_length})",
            max_sequence_size,
        );

        assert!(
            d_model_input == d_model,
            "d_model({}) of the input must be equal to d_model of encoding({})",
            d_model_input,
            d_model,
        );

        let slices = [0..batch_size, 0..seq_length, 0..d_model];

        input.add(self.sinusoids.clone().slice(slices))
    }
}

/// Returns sinusoids for positional embedding introduced in
/// [Attention is all you need](https://arxiv.org/abs/1706.03762).
///
/// The reference implementation can be found here:
/// [LANGUAGE MODELING WITH NN.TRANSFORMER AND TORCHTEXT
/// ](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
///
/// # Arguments
///
/// * `length` - The length of the sequence.
/// * `d_model` - The size of each vector.
/// * `max_timescale` - The maximum time scale to use.
///
/// # Returns
///
/// A tensor of shape [length, d_model] containing the sinusoids.
pub fn generate_sinusoids<B: Backend>(
    length: usize,
    d_model: usize,
    max_timescale: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    assert!(d_model % 2 == 0, "d_model must be even");
    assert!(
        max_timescale >= length,
        "max_timescale must be greater than length"
    );

    // Calculate the increment for the logarithmic timescale
    let log_timescale_increment = -(max_timescale as f32).ln() / d_model as f32;

    // Create a vector to hold the sinusoids
    let mut scaled_time_sin_cos = Vec::with_capacity(length);

    // Loop over each position in the sequence
    for i in 0..length {
        // Create a vector to hold the sinusoids for this position
        let mut row = Vec::with_capacity(d_model / 2);
        // Loop over each dimension of the sinusoids
        for k in (0..d_model).step_by(2) {
            // Calculate the division term for this dimension
            let div_term = (k as f32 * log_timescale_increment).exp();
            // Calculate the sine and cosine values for this dimension and position
            row.push((div_term * i as f32).sin());
            row.push((div_term * i as f32).cos());
        }

        // Add the sinusoids for this position to the vector
        scaled_time_sin_cos.push(row);
    }

    // Convert the sinusoids to a tensor and return it
    let data = Data::new(
        scaled_time_sin_cos.into_iter().flatten().collect(),
        [length, d_model].into(),
    );

    Tensor::<B, 2>::from_data(data.convert(), device)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_module() {
        let d_model = 6;
        let length = 3;

        // expected to broadcast
        let batch_size = 2;

        let device = Default::default();
        let pe = PositionalEncodingConfig::new(d_model).init::<TestBackend>(&device);

        // Use a tensor of zeros as input for easy verification of the output
        // The output should be the sinusoids broadcasted to the input shape
        let tensor = Tensor::zeros([batch_size, length, d_model], &device);

        let output = pe.forward(tensor);

        assert_eq!(output.shape().dims, [batch_size, length, d_model]);

        let expected = Tensor::<TestBackend, 3>::from_floats(
            [
                [
                    [0.00000, 1.00000, 0.00000, 1.00000, 0.00000, 1.00000],
                    [0.84147, 0.54030, 0.04640, 0.99892, 0.00215, 1.00000],
                    [0.90930, -0.41615, 0.09270, 0.99569, 0.00431, 0.99999],
                ],
                [
                    [0.00000, 1.00000, 0.00000, 1.00000, 0.00000, 1.00000],
                    [0.84147, 0.54030, 0.04640, 0.99892, 0.00215, 1.00000],
                    [0.90930, -0.41615, 0.09270, 0.99569, 0.00431, 0.99999],
                ],
            ],
            &device,
        );

        output.to_data().assert_approx_eq(&expected.to_data(), 5);
    }

    #[test]
    fn test_generate_sinusoids() {
        let device = Default::default();
        let sinusoids = generate_sinusoids::<TestBackend>(12, 6, 10_000, &device);

        // The values are taken from the pytorch reference implementation
        let expected = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.00000, 1.00000, 0.00000, 1.00000, 0.00000, 1.00000],
                [0.84147, 0.54030, 0.04640, 0.99892, 0.00215, 1.00000],
                [0.90930, -0.41615, 0.09270, 0.99569, 0.00431, 0.99999],
                [0.14112, -0.98999, 0.13880, 0.99032, 0.00646, 0.99998],
                [-0.75680, -0.65364, 0.18460, 0.98281, 0.00862, 0.99996],
                [-0.95892, 0.28366, 0.23000, 0.97319, 0.01077, 0.99994],
                [-0.27942, 0.96017, 0.27491, 0.96147, 0.01293, 0.99992],
                [0.65699, 0.75390, 0.31922, 0.94768, 0.01508, 0.99989],
                [0.98936, -0.14550, 0.36285, 0.93185, 0.01723, 0.99985],
                [0.41212, -0.91113, 0.40570, 0.91401, 0.01939, 0.99981],
                [-0.54402, -0.83907, 0.44767, 0.89420, 0.02154, 0.99977],
                [-0.99999, 0.00443, 0.48868, 0.87246, 0.02370, 0.99972],
            ],
            &device,
        );
        sinusoids.to_data().assert_approx_eq(&expected.to_data(), 5);
    }

    #[test]
    #[should_panic]
    fn d_model_input_should_match() {
        let d_model = 8;
        let device = Default::default();
        let pe = PositionalEncodingConfig::new(d_model).init::<TestBackend>(&device);
        let input = Tensor::zeros([1, 5, 10], &device);
        let _output = pe.forward(input);
    }

    #[test]
    #[should_panic]
    fn input_length_should_be_less_than_max_len() {
        let d_model = 8;
        let device = Default::default();
        let pe = PositionalEncodingConfig::new(d_model).init::<TestBackend>(&device);
        let input = Tensor::zeros([1, 6_000, d_model], &device);
        let _output = pe.forward(input);
    }
}
