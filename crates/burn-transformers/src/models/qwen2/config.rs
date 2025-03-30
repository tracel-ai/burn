use burn::config::Config;

/// Configuration for the Qwen2 model.
///
/// This configuration struct holds hyperparameters for the Qwen2 model architecture.
/// It is designed to be compatible with configurations from the Hugging Face `transformers` library.
#[derive(Config, Debug)]
pub struct Qwen2Config {
    /// The size of the vocabulary.
    pub vocab_size: usize,
    /// The hidden size of the model.
    pub hidden_size: usize,
    /// The number of intermediate layers in the MLP block.
    pub intermediate_size: usize,
    /// The number of hidden layers in the model.
    pub num_hidden_layers: usize,
    /// The number of attention heads.
    pub num_attention_heads: usize,
    /// The number of key/value heads for Grouped Query Attention (GQA).
    pub num_key_value_heads: usize,
    /// The maximum position embeddings sequence length.
    pub max_position_embeddings: usize,
    /// The maximum window size for sliding window attention.
    #[config(default = "4096")]
    pub sliding_window: usize,
    /// Epsilon value for RMS normalization.
    #[config(default = "1e-6")]
    pub rms_norm_eps: f64,
    /// Base value for RoPE (Rotary Positional Embedding).
    #[config(default = "10000.0")]
    pub rope_theta: f64,
    /// Whether to use bias in linear layers.
    #[config(default = "true")]
    pub no_bias: bool,
    /// The dropout probability for hidden layers.
    #[config(default = "0.0")]
    pub hidden_dropout: f64,
    /// The dropout probability for attention layers.
    #[config(default = "0.0")]
    pub attention_dropout: f64,
    /// Partial Rotary Factor
    #[config(default = "0.5")]
    pub partial_rotary_factor: f64,
    /// Tie word embeddings and language model head weights.
    #[config(default = "true")]
    pub tie_word_embeddings: bool,
    /// Path to the tokenizer
    #[config(default = "String::new()")]
    pub tokenizer: String,
}

// Default values based on common Qwen2 configurations

fn default_sliding_window() -> usize {
    4096 // Or a large value if sliding window is not used by default
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_rope_theta() -> f64 {
    10000.0
}

fn default_no_bias() -> bool {
    true // Qwen2 typically doesn't use bias
}

fn default_partial_rotary_factor() -> f64 {
    0.5 // Example, adjust if needed based on specific Qwen2 versions
}

fn default_tie_word_embeddings() -> bool {
    true
}

impl Qwen2Config {
    /// Initializes a new Qwen2Config with default values for a specific model size (e.g., 7B).
    /// Values should be adjusted based on the target model.
    pub fn init_default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 4096,
            intermediate_size: 11008, // Often 2.75 * hidden_size, check specific model
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32, // Set equal to num_attention_heads for MHA
            max_position_embeddings: 4096,
            sliding_window: default_sliding_window(),
            rms_norm_eps: default_rms_norm_eps(),
            rope_theta: default_rope_theta(),
            no_bias: default_no_bias(),
            hidden_dropout: 0.0,
            attention_dropout: 0.0,
            partial_rotary_factor: default_partial_rotary_factor(),
            tie_word_embeddings: default_tie_word_embeddings(),
            tokenizer: String::new(),
        }
    }
}

#[test]
fn test_qwen2_config() {
    // Create a tiny configuration for testing
    let config = Qwen2Config {
        vocab_size: 1000,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 4, // No GQA for simplicity
        max_position_embeddings: 128,
        sliding_window: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        no_bias: true,
        hidden_dropout: 0.0,
        attention_dropout: 0.0,
        partial_rotary_factor: 0.5,
        tie_word_embeddings: true,
        tokenizer: "dummy".to_string(),
    };

    // Just check that config is valid
    assert_eq!(config.vocab_size, 1000);
    assert_eq!(config.hidden_size, 128);
    assert_eq!(config.num_attention_heads, 4);
    assert_eq!(config.num_key_value_heads, 4);
}
