use burn::{
    module::Module,
    nn::{RotaryEncoding, RotaryEncodingConfig},
    record::{FileRecorder, RecorderError},
    tensor::{backend::Backend, Device, Int, Shape, Tensor, TensorData},
};

use crate::{
    tokenizer::Tokenizer,
    transformer::{KeyValueCache, Transformer, TransformerConfig},
};

pub mod config;
pub use config::Qwen2Config;

/// Generated text sample output.
pub struct GenerationOutput {
    /// The generated text.
    pub text: String,
    /// The number of generated tokens.
    pub tokens: usize,
    /// The time it took to produce the output tokens (generation + decoding).
    pub time: f64,
}

/// Qwen2 large language model with tokenizer.
pub struct Qwen2<B: Backend, T: Tokenizer> {
    /// The tokenizer.
    pub tokenizer: T,
    /// Qwen2 decoder-only transformer.
    pub model: Transformer<B>,
    /// Key-value cache for each transformer block.
    pub cache: Vec<KeyValueCache<B>>,
    /// Rotary positional encoding (RoPE).
    pub rope: RotaryEncoding<B>,
    pub device: Device<B>,
}

impl Qwen2Config {
    /// Initialize a new Qwen2 model.
    pub fn init<B: Backend, T: Tokenizer>(
        &self,
        device: &Device<B>,
    ) -> Result<Qwen2<B, T>, String> {
        let tokenizer = T::new(&self.tokenizer)?;

        // Map configuration to TransformerConfig
        let model = TransformerConfig {
            vocab_size: self.vocab_size,
            n_layers: self.num_hidden_layers,
            d_model: self.hidden_size,
            hidden_size: self.intermediate_size,
            n_heads: self.num_attention_heads,
            n_kv_heads: self.num_key_value_heads,
            max_seq_len: self.max_position_embeddings,
            norm_eps: self.rms_norm_eps,
        }
        .init(device);

        // Initialize caches for each layer
        let cache = (0..self.num_hidden_layers)
            .map(|_| {
                KeyValueCache::new(
                    1, // Use default batch size of 1
                    self.num_key_value_heads,
                    self.max_position_embeddings,
                    self.hidden_size / self.num_attention_heads,
                    device,
                )
            })
            .collect::<Vec<_>>();

        // Initialize rotary positional encoding
        let head_dim = self.hidden_size / self.num_attention_heads;
        let rotary_dim = (head_dim as f64 * self.partial_rotary_factor) as usize;

        let rope = RotaryEncodingConfig::new(self.max_position_embeddings, rotary_dim)
            .with_theta(self.rope_theta as f32)
            .init(device);

        Ok(Qwen2 {
            tokenizer,
            model,
            cache,
            rope,
            device: device.clone(),
        })
    }
}

impl<B: Backend, T: Tokenizer> Qwen2<B, T> {
    /// Forward pass through the model
    pub fn forward(
        &mut self,
        input_ids: Tensor<B, 2, Int>,
        _position_ids: Option<Tensor<B, 1, Int>>,
    ) -> Tensor<B, 3> {
        self.model.forward(input_ids, &mut self.cache, &self.rope)
    }

    /// Generate text sample based on the provided prompt.
    ///
    /// # Arguments
    /// - `prompt`: The prompt string to use for generating the samples.
    /// - `sample_len`: The number of new tokens to generate.
    /// - `temperature`: Temperature for sampling (scales logits).
    /// - `sampler`: The sampling strategy for next token selection.
    ///
    /// # Returns
    /// The generated text along with metadata.
    pub fn generate(
        &mut self,
        prompt: &str,
        sample_len: usize,
        temperature: f64,
        sampler: &mut crate::sampling::Sampler,
    ) -> Result<GenerationOutput, String> {
        use burn::tensor::ElementConversion;
        use std::time::Instant;

        // Tokenize the prompt
        let tokens = self.tokenizer.encode(prompt, false, false);
        let prompt_len = tokens.len();

        // Create tensor for input and output tokens
        let mut output_tokens = Tensor::<B, 1, Int>::empty([prompt_len + sample_len], &self.device);
        let shape = Shape::new([tokens.len()]);
        let input_tokens =
            Tensor::<B, 1, Int>::from_data(TensorData::new(tokens, shape), &self.device);
        output_tokens = output_tokens.slice_assign([0..prompt_len], input_tokens);

        // Stop tokens
        let stop_tokens = Tensor::from_ints(self.tokenizer.stop_ids().as_slice(), &self.device);

        // Generation
        let mut num_tokens: usize = 0;
        let mut input_pos = Tensor::<B, 1, Int>::arange(0..prompt_len as i64, &self.device);
        let now = Instant::now();

        for i in 0..sample_len {
            let x = output_tokens
                .clone()
                .select(0, input_pos.clone())
                .reshape([1, -1]);
            let logits = self.forward(x, None);

            let [batch_size, seq_len, _vocab_size] = logits.dims();
            let mut next_token_logits = logits
                .slice([0..batch_size, seq_len - 1..seq_len])
                .squeeze(1);

            // Apply temperature if needed
            if temperature > 0.0 {
                next_token_logits = crate::models::llama::temperature_scaled_softmax(
                    next_token_logits,
                    temperature,
                );
            }

            // Sample next token
            let next_token = sampler.sample(next_token_logits).squeeze(0);

            // Stop if we encounter a stop token
            if stop_tokens
                .clone()
                .equal(next_token.clone())
                .any()
                .into_scalar()
                .elem()
            {
                break;
            }

            // Update with generated token
            output_tokens =
                output_tokens.slice_assign([prompt_len + i..prompt_len + i + 1], next_token);
            num_tokens += 1;

            // Advance position
            let t = input_pos.dims()[0];
            input_pos = input_pos.slice([t - 1..t]) + 1;
        }

        // Extract generated tokens
        let tokens = output_tokens
            .into_data()
            .as_slice::<B::IntElem>()
            .map_err(|e| format!("Failed to convert tokens to slice: {:?}", e))?
            [prompt_len..prompt_len + num_tokens]
            .iter()
            .map(|t| t.elem::<u32>())
            .collect::<Vec<_>>();

        // Decode tokens to text
        let generated = self.tokenizer.decode(tokens)?;
        let elapsed = now.elapsed().as_secs_f64();

        Ok(GenerationOutput {
            text: generated,
            tokens: num_tokens,
            time: elapsed,
        })
    }

    /// Reset the model state (used between generations)
    pub fn reset(&mut self) {
        self.cache.iter_mut().for_each(|cache| cache.reset());
    }

    /// Load model from file using the specified recorder
    pub fn load<R: FileRecorder<B>>(
        mut self,
        file_path: &str,
        recorder: &R,
    ) -> Result<Self, RecorderError> {
        self.model = self.model.load_file(file_path, recorder, &self.device)?;
        Ok(self)
    }

    /// Save model to file using the specified recorder
    pub fn save<R: FileRecorder<B>>(
        self,
        file_path: &str,
        recorder: &R,
    ) -> Result<(), RecorderError> {
        self.model.save_file(file_path, recorder)
    }
}
