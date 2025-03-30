use burn::{
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNormConfig, RotaryEncoding,
        RotaryEncodingConfig,
    },
    tensor::{
        activation::{self, softmax},
        backend::Backend,
        Int, Tensor,
    },
};

use crate::models::qwen2::config::Qwen2Config;

#[derive(Module, Debug)]
pub struct Qwen2Attention<B: Backend> {
    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    rotary_emb: RotaryEncoding<B>,
    hidden_size: usize,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

impl<B: Backend> Qwen2Attention<B> {
    pub fn new(config: &Qwen2Config, device: &B::Device) -> Self {
        let head_dim = config.hidden_size / config.num_attention_heads;

        let q_proj = LinearConfig::new(config.hidden_size, config.num_attention_heads * head_dim)
            .with_bias(!config.no_bias)
            .init(device);

        let k_proj = LinearConfig::new(config.hidden_size, config.num_key_value_heads * head_dim)
            .with_bias(!config.no_bias)
            .init(device);

        let v_proj = LinearConfig::new(config.hidden_size, config.num_key_value_heads * head_dim)
            .with_bias(!config.no_bias)
            .init(device);

        let o_proj = LinearConfig::new(config.num_attention_heads * head_dim, config.hidden_size)
            .with_bias(!config.no_bias)
            .init(device);

        // Create rotary positional embedding
        // For Qwen2, we'll use partial rotary embeddings (only apply to part of the head dimension)
        let rotary_dim = (head_dim as f64 * config.partial_rotary_factor) as usize;
        let rotary_emb = RotaryEncodingConfig::new(config.max_position_embeddings, rotary_dim * 2)
            .with_theta(config.rope_theta as f32)
            .init(device);

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            hidden_size: config.hidden_size,
            num_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim,
        }
    }

    // Helper function to repeat key/value heads for grouped query attention
    fn repeat_kv(&self, hidden_states: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
        let [batch_size, num_kv_heads, seq_len, head_dim] = hidden_states.dims();

        hidden_states
            .reshape([batch_size, num_kv_heads, 1, seq_len, head_dim])
            .repeat(&[1, 1, n_rep, 1, 1])
            .reshape([batch_size, num_kv_heads * n_rep, seq_len, head_dim])
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        position_ids: Option<Tensor<B, 1, Int>>,
        attention_mask: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();

        let device = x.device();

        // Project queries, keys, and values
        let query = self.q_proj.forward(x.clone());
        let key = self.k_proj.forward(x.clone());
        let value = self.v_proj.forward(x);

        // Reshape for multi-head attention
        let query = query.reshape([batch_size, seq_len, self.num_heads, self.head_dim]);
        let key = key.reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim]);
        let value = value.reshape([batch_size, seq_len, self.num_key_value_heads, self.head_dim]);

        // In actual Qwen2 implementation, position_ids would be used for rotary embeddings
        // Since our RotaryEncoding.apply only takes an offset, we'll continue using 0
        // But we'll keep the position_ids generation in case a future implementation needs it
        let _position_ids = match position_ids {
            Some(pos) => pos,
            None => Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device),
        };

        // Transpose tensors for attention calculation - be careful with dimensions for ndarray backend
        // In ndarray backend, specific dimension ordering is needed for matmul operations
        // Using explicit reshape operations instead of permute when dealing with the ndarray backend
        let mut query_states = Vec::new();
        let mut key_states = Vec::new();
        let mut value_states = Vec::new();

        // Process each batch individually to maintain proper dimension order
        for b in 0..batch_size {
            let batch_query =
                query
                    .clone()
                    .slice([b..b + 1, 0..seq_len, 0..self.num_heads, 0..self.head_dim]);
            let batch_key = key.clone().slice([
                b..b + 1,
                0..seq_len,
                0..self.num_key_value_heads,
                0..self.head_dim,
            ]);
            let batch_value = value.clone().slice([
                b..b + 1,
                0..seq_len,
                0..self.num_key_value_heads,
                0..self.head_dim,
            ]);

            // Reshape to [1, num_heads, seq_len, head_dim]
            let reshaped_query = batch_query.reshape([1, self.num_heads, seq_len, self.head_dim]);
            let reshaped_key =
                batch_key.reshape([1, self.num_key_value_heads, seq_len, self.head_dim]);
            let reshaped_value =
                batch_value.reshape([1, self.num_key_value_heads, seq_len, self.head_dim]);

            query_states.push(reshaped_query);
            key_states.push(reshaped_key);
            value_states.push(reshaped_value);
        }

        // Concatenate along batch dimension
        let query_layer = Tensor::cat(query_states, 0);
        let key_layer = Tensor::cat(key_states, 0);
        let value_layer = Tensor::cat(value_states, 0);

        // Apply rotary embeddings
        let query_rot = self.rotary_emb.apply(query_layer, 0);
        let key_rot = self.rotary_emb.apply(key_layer, 0);

        // Handle grouped query attention if needed
        let (key_states, value_states) = if self.num_key_value_heads < self.num_heads {
            // Expand key/value heads to match num_heads for grouped query attention
            let key_states = self.repeat_kv(key_rot, self.num_heads / self.num_key_value_heads);
            let value_states =
                self.repeat_kv(value_layer, self.num_heads / self.num_key_value_heads);
            (key_states, value_states)
        } else {
            (key_rot, value_layer)
        };

        // Calculate attention scores and scale
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // Process for matrix multiplication with compatible dimensions
        let mut attention_scores_all = Vec::new();

        for b in 0..batch_size {
            let q = query_rot.clone().slice([
                b..b + 1,
                0..self.num_heads,
                0..seq_len,
                0..self.head_dim,
            ]);
            let k = key_states.clone().slice([
                b..b + 1,
                0..self.num_heads,
                0..seq_len,
                0..self.head_dim,
            ]);

            // Transpose key to [1, num_heads, head_dim, seq_len] for matmul
            let k_t = k.permute([0, 1, 3, 2]);

            // Compute attention scores [1, num_heads, seq_len, seq_len]
            let scores = q.matmul(k_t) * scale;
            attention_scores_all.push(scores);
        }

        // Combine all attention scores
        let attention_scores = Tensor::cat(attention_scores_all, 0);

        // Apply attention mask if provided
        let masked_attention_scores = if let Some(mask) = attention_mask {
            // Reshape mask for compatibility with attention scores
            let mut batch_masks = Vec::new();

            for b in 0..batch_size {
                let batch_mask = mask.clone().slice([b..b + 1, 0..seq_len, 0..seq_len]);
                // Reshape to [1, 1, seq_len, seq_len] and repeat for all heads
                let expanded = batch_mask.reshape([1, 1, seq_len, seq_len]);
                let mut head_masks = Vec::new();

                for _ in 0..self.num_heads {
                    head_masks.push(expanded.clone());
                }

                let combined = Tensor::cat(head_masks, 1);
                batch_masks.push(combined);
            }

            let mask_expanded = Tensor::cat(batch_masks, 0);

            // Apply mask: add large negative values (like -10000) to masked positions
            attention_scores + (mask_expanded - 1.0) * 10000.0
        } else {
            attention_scores
        };

        // Apply softmax
        let attention_probs = softmax(masked_attention_scores, 3);

        // Apply attention to values
        let mut context_all = Vec::new();

        for b in 0..batch_size {
            let probs = attention_probs.clone().slice([
                b..b + 1,
                0..self.num_heads,
                0..seq_len,
                0..seq_len,
            ]);
            let vals = value_states.clone().slice([
                b..b + 1,
                0..self.num_heads,
                0..seq_len,
                0..self.head_dim,
            ]);

            // Compute weighted sum [1, num_heads, seq_len, head_dim]
            let ctx = probs.matmul(vals);
            context_all.push(ctx);
        }

        let context = Tensor::cat(context_all, 0);

        // Reshape back to original dimensions with explicit operations to maintain order
        let mut reshaped_context_all = Vec::new();

        for b in 0..batch_size {
            let batch_context =
                context
                    .clone()
                    .slice([b..b + 1, 0..self.num_heads, 0..seq_len, 0..self.head_dim]);
            // Reshape to [1, seq_len, num_heads, head_dim]
            let reshaped = batch_context.permute([0, 2, 1, 3]).reshape([
                1,
                seq_len,
                self.num_heads * self.head_dim,
            ]);
            reshaped_context_all.push(reshaped);
        }

        let output = Tensor::cat(reshaped_context_all, 0);

        // Project to output dimension
        self.o_proj.forward(output)
    }
}

#[derive(Module, Debug)]
pub struct Qwen2MLP<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> Qwen2MLP<B> {
    pub fn new(config: &Qwen2Config, device: &B::Device) -> Self {
        // In Qwen2, the MLP consists of gate_proj and up_proj followed by down_proj
        let gate_proj = LinearConfig::new(config.hidden_size, config.intermediate_size)
            .with_bias(!config.no_bias)
            .init(device);

        let up_proj = LinearConfig::new(config.hidden_size, config.intermediate_size)
            .with_bias(!config.no_bias)
            .init(device);

        let down_proj = LinearConfig::new(config.intermediate_size, config.hidden_size)
            .with_bias(!config.no_bias)
            .init(device);

        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Implementation based on Qwen2 MLP architecture with SwiGLU activation
        let gate = activation::silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);

        let intermediate = gate * up;
        self.down_proj.forward(intermediate)
    }
}

// Fix the model to use QwenRmsNorm instead of RmsNorm to avoid name conflicts
#[derive(Module, Debug)]
pub struct QwenRmsNorm<B: Backend> {
    norm: burn::nn::RmsNorm<B>,
}

impl<B: Backend> QwenRmsNorm<B> {
    pub fn new(config: &Qwen2Config, device: &B::Device) -> Self {
        let norm = RmsNormConfig::new(config.hidden_size)
            .with_epsilon(config.rms_norm_eps)
            .init(device);

        Self { norm }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.norm.forward(x)
    }
}

// Update Qwen2DecoderLayer to use QwenRmsNorm
#[derive(Module, Debug)]
pub struct Qwen2DecoderLayer<B: Backend> {
    pub self_attn: Qwen2Attention<B>,
    pub mlp: Qwen2MLP<B>,
    pub input_layernorm: QwenRmsNorm<B>,
    pub post_attention_layernorm: QwenRmsNorm<B>,
}

impl<B: Backend> Qwen2DecoderLayer<B> {
    pub fn new(config: &Qwen2Config, device: &B::Device) -> Self {
        let self_attn = Qwen2Attention::new(config, device);
        let mlp = Qwen2MLP::new(config, device);

        let input_layernorm = QwenRmsNorm::new(config, device);
        let post_attention_layernorm = QwenRmsNorm::new(config, device);

        Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        position_ids: Option<Tensor<B, 1, Int>>,
        attention_mask: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        // Implementation based on Qwen2 decoder layer with RMSNorm
        let residual = x.clone();

        // Self attention
        let norm_x = self.input_layernorm.forward(x);
        let attn_output = self.self_attn.forward(norm_x, position_ids, attention_mask);
        let x = residual + attn_output;

        // MLP
        let residual = x.clone();
        let norm_x = self.post_attention_layernorm.forward(x);
        let mlp_output = self.mlp.forward(norm_x);

        residual + mlp_output
    }
}

// Add a RotaryEmbedding wrapper for test compatibility
#[derive(Module, Debug)]
pub struct RotaryEmbedding<B: Backend> {
    rotary: RotaryEncoding<B>,
    head_dim: usize,
}

impl<B: Backend> RotaryEmbedding<B> {
    pub fn new(config: &Qwen2Config, device: &B::Device) -> Self {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let rotary_dim = (head_dim as f64 * config.partial_rotary_factor) as usize;

        let rotary = RotaryEncodingConfig::new(config.max_position_embeddings, rotary_dim * 2)
            .with_theta(config.rope_theta as f32)
            .init(device);

        Self { rotary, head_dim }
    }

    pub fn forward(
        &mut self,
        x: Tensor<B, 3>,
        _seq_len: usize,
        _device: &B::Device,
    ) -> Tensor<B, 3> {
        // Reshape the input for rotary application
        // The input is [batch, seq, hidden_size], we need to reshape to [batch, seq, num_heads, head_dim]
        let [batch, seq, hidden] = x.dims();
        let num_heads = hidden / self.head_dim;

        let reshaped = x.reshape([batch, seq, num_heads, self.head_dim]);

        // Apply rotary embeddings
        let rotated = self.rotary.apply(reshaped.permute([0, 2, 1, 3]), 0);

        // Permute back and reshape to original shape
        rotated.permute([0, 2, 1, 3]).reshape([batch, seq, hidden])
    }
}

/// Qwen2 Model
#[derive(Module, Debug)]
pub struct Qwen2Model<B: Backend> {
    embed_tokens: Embedding<B>,
    layers: Vec<Qwen2DecoderLayer<B>>,
    norm: QwenRmsNorm<B>,
    lm_head: Linear<B>,
    config_hidden_size: usize,
    config_num_hidden_layers: usize,
    config_num_attention_heads: usize,
    config_vocab_size: usize,
}

impl<B: Backend> Qwen2Model<B> {
    /// Create a new Qwen2 model from a configuration
    pub fn new(config: Qwen2Config, device: &B::Device) -> Self {
        let embed_tokens = EmbeddingConfig::new(config.vocab_size, config.hidden_size).init(device);

        let layers: Vec<Qwen2DecoderLayer<B>> = (0..config.num_hidden_layers)
            .map(|_| Qwen2DecoderLayer::new(&config, device))
            .collect();

        let norm = QwenRmsNorm::new(&config, device);

        let lm_head = LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);

        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            config_hidden_size: config.hidden_size,
            config_num_hidden_layers: config.num_hidden_layers,
            config_num_attention_heads: config.num_attention_heads,
            config_vocab_size: config.vocab_size,
        }
    }

    /// Forward pass through the model
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        position_ids: Option<Tensor<B, 1, Int>>,
        attention_mask: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let x = self.embed_tokens.forward(input_ids);

        let mut hidden_states = x;

        // Apply decoder layers
        for layer in &self.layers {
            hidden_states =
                layer.forward(hidden_states, position_ids.clone(), attention_mask.clone());
        }

        // Final normalization
        let hidden_states = self.norm.forward(hidden_states);

        // Language model head
        self.lm_head.forward(hidden_states)
    }
}

// Now implement Qwen2ForCausalLM
#[derive(Module, Debug)]
pub struct Qwen2ForCausalLM<B: Backend> {
    model: Qwen2Model<B>,
}

impl<B: Backend> Qwen2ForCausalLM<B> {
    pub fn new(config: &Qwen2Config, device: &B::Device) -> Self {
        let model = Qwen2Model::new(config.clone(), device);
        Self { model }
    }

    pub fn forward(
        &mut self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2>>,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        // Get batch size and sequence length from input_ids
        let [batch_size, seq_len] = input_ids.dims();

        // Generate position IDs if not provided
        // For causal LM, position IDs should just be sequential from 0 to seq_len-1
        let position_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device);

        // Process attention mask if provided
        let processed_attention_mask = if let Some(mask) = attention_mask {
            // Reshape from [batch_size, seq_len*seq_len] to [batch_size, seq_len, seq_len]
            // This fixes the dimension order for ndarray backend

            // First check if the mask is just a causal mask from create_causal_mask
            if mask.dims()[1] == seq_len * seq_len {
                // This is a flattened mask, reshape it to 3D
                let mut batch_masks = Vec::new();

                for b in 0..batch_size {
                    // Get this batch's mask
                    let batch_mask = mask.clone().slice([b..b + 1, 0..seq_len * seq_len]);
                    // Reshape to 3D
                    let reshaped = batch_mask.reshape([1, seq_len, seq_len]);
                    batch_masks.push(reshaped);
                }

                Some(Tensor::cat(batch_masks, 0))
            } else {
                // This is a regular padding mask [batch_size, seq_len]
                // Create a 3D attention mask for the transformer model
                // First, create a causal mask where positions can attend to previous positions
                let seq = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device);
                let row_indices = seq.clone().reshape([seq_len, 1]);
                let col_indices = seq.reshape([1, seq_len]);
                let causal_mask = row_indices.greater_equal(col_indices).float();

                // Process each batch item to create 3D masks
                let mut batch_masks = Vec::new();

                for b in 0..batch_size {
                    // Extract this batch's mask [1, seq_len]
                    let batch_mask = mask.clone().slice([b..b + 1, 0..seq_len]);

                    // Reshape to [1, 1, seq_len] and broadcast across rows
                    let expanded = batch_mask.reshape([1, 1, seq_len]).repeat(&[1, seq_len, 1]);

                    // Combine with causal mask
                    let combined = expanded * causal_mask.clone().reshape([1, seq_len, seq_len]);
                    batch_masks.push(combined);
                }

                Some(Tensor::cat(batch_masks, 0))
            }
        } else {
            // If no explicit mask provided, create a causal mask
            let seq = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device);
            let row_indices = seq.clone().reshape([seq_len, 1]);
            let col_indices = seq.reshape([1, seq_len]);

            // Create mask [batch_size, seq_len, seq_len] by building a batch of masks
            let causal_mask = row_indices
                .greater_equal(col_indices)
                .float()
                .reshape([1, seq_len, seq_len]);

            let mut batch_masks = Vec::new();
            for _ in 0..batch_size {
                batch_masks.push(causal_mask.clone());
            }

            Some(Tensor::cat(batch_masks, 0))
        };

        // Forward pass through the model
        self.model
            .forward(input_ids, Some(position_ids), processed_attention_mask)
    }

    pub fn create_causal_mask(
        &self,
        batch_size: usize,
        seq_len: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        // In tests, the mask needs to be [batch_size, seq_len * seq_len] with the correct dimension order
        // First create a sequence [0, 1, 2, ..., seq_len-1]
        let seq = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device);

        // Create row and column indices
        let row_indices = seq.clone().reshape([seq_len, 1]);
        let col_indices = seq.reshape([1, seq_len]);

        // Create causal mask [seq_len, seq_len] where positions can attend to themselves and previous positions
        // This creates a lower triangular matrix with 1s for valid positions
        let mask = row_indices.greater_equal(col_indices).float();

        // Reshape to match what the tests expect: [batch_size, seq_len*seq_len]
        // First reshape to [1, seq_len*seq_len]
        let flat_mask = mask.reshape([1, seq_len * seq_len]);

        // If batch size is 1, we can just return the flat mask
        if batch_size == 1 {
            return flat_mask;
        }

        // For batch_size > 1, we need to tile/repeat the mask for each batch item
        // But to avoid dimension mismatch issues with ndarray, we'll manually stack the masks
        let mut batch_masks = Vec::new();
        for _ in 0..batch_size {
            batch_masks.push(flat_mask.clone());
        }

        // Stack along batch dimension
        Tensor::cat(batch_masks, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen2_config() {
        // Test that we can create a valid Qwen2Config
        let config = Qwen2Config {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 512,
            sliding_window: 512,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            no_bias: true,
            hidden_dropout: 0.0,
            attention_dropout: 0.0,
            partial_rotary_factor: 0.5,
            tie_word_embeddings: true,
        };

        assert_eq!(config.vocab_size, 1000);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.num_attention_heads, 4);
        assert_eq!(config.num_key_value_heads, 2);
    }
}
