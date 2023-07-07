use alloc::vec::Vec;

use crate as burn;
use crate::config::Config;
use crate::module::Module;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use burn_tensor::Data;

use libm::{cosf, expf, logf, sinf};

/// Configuration to create an [SinEmbedding](SinEmbedding) layer.
#[derive(Config)]
pub struct SinEmbeddingConfig {
    /// The number of embedding vectors.
    n_embedding: usize,

    /// The size of each vector.
    d_model: usize,

    /// Max time scale to use.
    #[config(default = "10_000")]
    max_timescale: usize,
}

/// Lookup table to store sinusoidal vectors for positional embedding.
#[derive(Module, Debug)]
pub struct SinEmbedding<B: Backend> {
    sinusoids: Tensor<B, 3>,
}

impl SinEmbeddingConfig {
    /// Initialize a new [SinEmbedding](SinEmbedding) module.
    pub fn init<B: Backend>(&self) -> SinEmbedding<B> {
        let sinusoids = generate_sinusoids::<B>(self.n_embedding, self.d_model, self.max_timescale)
            .unsqueeze::<3>();

        SinEmbedding { sinusoids }
    }
}

impl<B: Backend> SinEmbedding<B> {
    /// Applies the forward pass on the input tensor by adding the sinusoids to the input.
    ///
    /// # Shapes
    ///
    /// - input: [batch_size, seq_length, d_model]
    /// - output: [batch_size, seq_length, d_model]
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        input.add(self.sinusoids.clone())
    }
}

/// Returns sinusoids for positional embedding introduced in [Attention is all you need
/// ](https://arxiv.org/abs/1706.03762).
///
/// The sinusoids are calculated by multiplying the sequence length by the number of channels
/// and then generating a sequence of numbers from 0 to this product. This sequence is then
/// multiplied by a factor that is calculated by taking the natural logarithm of the maximum
/// timescale and dividing it by the number of channels divided by 2 minus 1. This factor is
/// then applied to the exponential function, resulting in a sequence of numbers that are
/// scaled by a factor that increases exponentially. This sequence is then multiplied by
/// the sine and cosine functions, resulting in two arrays of the same shape. These two arrays
/// are then concatenated along the second dimension, resulting in an array of shape
/// (sequence_length, channels).
///
/// The reference implementation is taken from [Whisper's sinusoids function](
/// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L53).
///
///
/// # Arguments
///
/// * `length` - The length of the sequence
/// * `d_model` - The number of channels
/// * `max_timescale` - The maximum timescale
///
/// # Example
///
/// ```
/// use burn_ndarray::NdArrayBackend;
/// use burn_core::nn::generate_sinusoids;
///
/// type TestBackend = NdArrayBackend<f32>;
///
/// fn main() {
///    let sinusoids = generate_sinusoids::<TestBackend>(8, 4, 128);
///    println!("{}", sinusoids);
/// }
///
/// ```
pub fn generate_sinusoids<B: Backend>(
    length: usize,
    d_model: usize,
    max_timescale: usize,
) -> Tensor<B, 2> {
    // input checks
    assert!(d_model % 2 == 0, "d_model must be even");
    assert!(
        max_timescale >= length,
        "max_timescale must be greater than length"
    );

    // calculate log_timescale_increment by taking the natural logarithm of max_timescale
    // and dividing it by d_model / 2 - 1.
    let log_timescale_increment = logf(max_timescale as f32) / ((d_model / 2) - 1) as f32;

    // create an array of inverse timescales, inv_timescales, by generating a sequence of numbers
    // from 0 to d_model / 2 - 1, multiplying each by -log_timescale_increment, and applying
    // the exponential function to each.
    let inv_timescales = (0..d_model / 2)
        .map(|x| expf(-log_timescale_increment * x as f32))
        .collect::<Vec<_>>();

    // The scaled_time is calculated by multiplying inv_timescales with an array of numbers
    // from 0 to length - 1. This operation performs element-wise multiplication between
    // each number in the sequence from 0 to length - 1 and the inv_timescales array.
    // This results in a 2D array where each column is a sequence of numbers from 0 to length - 1,
    // scaled by a different factor from inv_timescales.
    let scaled_time = (0..length)
        .map(|i| {
            inv_timescales
                .iter()
                .map(|x| x * i as f32)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Calculate the sine and cosine of each element in the scaled_time array,
    // resulting in two arrays of the same shape.
    let scaled_time_sin = scaled_time
        .iter()
        .map(|x| x.iter().map(|y| sinf(*y)).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    let scaled_time_cos = scaled_time
        .iter()
        .map(|x| x.iter().map(|y| cosf(*y)).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    // concatenate these two arrays along the second dimension
    // (i.e., it adds the second array as new columns to the first array)
    let scaled_time_sin_cos = scaled_time_sin
        .into_iter()
        .zip(scaled_time_cos.into_iter())
        .map(|(mut x, mut y)| {
            x.append(&mut y);
            x
        })
        .collect::<Vec<_>>();

    // convert the result to a Tensor data
    let data = Data::new(
        scaled_time_sin_cos.into_iter().flatten().collect(),
        [length, d_model].into(),
    );

    // convert the data to a Tensor and return it
    Tensor::<B, 2>::from_data(data.convert())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_module() {
        let sin_embedding = SinEmbeddingConfig::new(8, 4)
            .with_max_timescale(128)
            .init::<TestBackend>();

        let tensor = Tensor::zeros([1, 8, 4]);

        let output = sin_embedding.forward(tensor);

        assert_eq!(output.shape().dims, [1usize, 8, 4]);

        // The values are taken from the reference implementation
        let expected = Tensor::<TestBackend, 2>::from_floats([
            [0.0, 0.0, 1.0, 1.0],
            [0.841471, 0.0078124204, 0.54030234, 0.9999695],
            [0.9092974, 0.015624364, -0.41614684, 0.9998779],
            [0.14112, 0.023435354, -0.9899925, 0.99972534],
            [-0.7568025, 0.031244913, -0.6536436, 0.9995118],
            [-0.9589243, 0.03905257, 0.2836622, 0.9992372],
            [-0.2794155, 0.046857838, 0.96017027, 0.99890155],
            [0.6569866, 0.054660246, 0.75390226, 0.998505],
        ]);

        output
            .squeeze(0)
            .to_data()
            .assert_approx_eq(&expected.to_data(), 7);
    }

    #[test]
    fn test_generate_sinusoids() {
        let sinusoids = generate_sinusoids::<TestBackend>(8, 4, 128);

        // The values are taken from the reference implementation
        let expected = Tensor::<TestBackend, 2>::from_floats([
            [0.0, 0.0, 1.0, 1.0],
            [0.841471, 0.0078124204, 0.54030234, 0.9999695],
            [0.9092974, 0.015624364, -0.41614684, 0.9998779],
            [0.14112, 0.023435354, -0.9899925, 0.99972534],
            [-0.7568025, 0.031244913, -0.6536436, 0.9995118],
            [-0.9589243, 0.03905257, 0.2836622, 0.9992372],
            [-0.2794155, 0.046857838, 0.96017027, 0.99890155],
            [0.6569866, 0.054660246, 0.75390226, 0.998505],
        ]);
        sinusoids.to_data().assert_approx_eq(&expected.to_data(), 7);
    }
}
