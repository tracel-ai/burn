use std::collections::HashMap;
use std::path::PathBuf;

use crate::gguf::{GGUFArchitecture, GGUFModel, GGUFValue};
use crate::tokenizer::Tokenizer;

/// Token information including the token string, score, and type
#[derive(Debug, Clone)]
pub struct TokenInfo {
    /// The token string
    pub token: String,
    /// Token score (used for sorting in some tokenizers)
    pub score: f32,
    /// Token type: 0=normal, 1=special, 2=user-defined, etc.
    pub token_type: u32,
}

/// GGUF tokenizer implementation for models in GGUF format
pub struct GGUFTokenizer {
    /// The vocabulary with token information
    pub vocabulary: Vec<TokenInfo>,
    /// Reverse mapping from token to ID
    pub token_to_id: HashMap<String, u32>,
    /// Beginning of sequence token ID
    pub bos_id: u32,
    /// End of sequence token ID
    pub eos_id: u32,
    /// Unknown token ID (optional)
    pub unk_id: Option<u32>,
    /// The architecture of the tokenizer
    pub architecture: GGUFArchitecture,
    /// Path to the GGUF model
    pub model_path: PathBuf,
}

impl GGUFTokenizer {
    /// Create a new GGUFTokenizer from a GGUF model
    fn from_gguf_model(model: &GGUFModel) -> Result<Self, String> {
        // Extract architecture first for better error handling
        let architecture = model.get_architecture();

        // Extract vocabulary from the model using multiple possible metadata keys
        let vocabulary = Self::extract_vocabulary(model)?;
        println!("Extracted vocabulary with {} tokens", vocabulary.len());

        // Build token to ID map
        let token_to_id: HashMap<String, u32> = vocabulary
            .iter()
            .enumerate()
            .map(|(id, token_info)| (token_info.token.clone(), id as u32))
            .collect();

        // Extract special token IDs
        let (bos_id, eos_id, unk_id) = Self::extract_special_tokens(model, &architecture);
        println!(
            "Special tokens - BOS: {}, EOS: {}, UNK: {:?}",
            bos_id, eos_id, unk_id
        );

        let model_path = model.file_path.clone();

        Ok(Self {
            vocabulary,
            token_to_id,
            bos_id,
            eos_id,
            unk_id,
            architecture,
            model_path,
        })
    }

    /// Extract vocabulary from model metadata using various possible keys
    fn extract_vocabulary(model: &GGUFModel) -> Result<Vec<TokenInfo>, String> {
        // Try using the enhanced typed vocabulary method
        if let Some(typed_vocab) = model.get_typed_vocabulary() {
            let vocabulary = typed_vocab
                .into_iter()
                .map(|entry| TokenInfo {
                    token: entry.token,
                    score: entry.score,
                    token_type: entry.token_type,
                })
                .collect();
            return Ok(vocabulary);
        }

        // Fallback: try using token_type_ids when available
        if let Some(GGUFValue::Array(token_type_ids)) =
            model.get_metadata("tokenizer.ggml.token_type_ids")
        {
            if let Some(GGUFValue::Array(tokens)) = model.get_metadata("tokenizer.ggml.tokens") {
                // Both tokens and token_type_ids exist
                let mut vocabulary = Vec::with_capacity(tokens.len());

                for (i, token_value) in tokens.iter().enumerate() {
                    if let GGUFValue::String(token_str) = token_value {
                        // Get token type ID if available, default to 0 (normal token)
                        let token_type = if i < token_type_ids.len() {
                            match &token_type_ids[i] {
                                GGUFValue::U32(id) => *id,
                                GGUFValue::I32(id) => *id as u32,
                                _ => 0,
                            }
                        } else {
                            0
                        };

                        // Get score if available
                        let score = if let Some(GGUFValue::Array(scores)) =
                            model.get_metadata("tokenizer.ggml.scores")
                        {
                            if i < scores.len() {
                                if let GGUFValue::F32(score_val) = scores[i] {
                                    score_val
                                } else {
                                    0.0
                                }
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };

                        vocabulary.push(TokenInfo {
                            token: token_str.clone(),
                            score,
                            token_type,
                        });
                    }
                }

                if !vocabulary.is_empty() {
                    return Ok(vocabulary);
                }
            }
        }

        // Fallback: try getting vocabulary directly from model method
        if let Some(vocab_tuples) = model.get_vocabulary() {
            let vocabulary = vocab_tuples
                .into_iter()
                .map(|(token, score)| TokenInfo {
                    token,
                    score,
                    token_type: 0, // Default to normal token type
                })
                .collect();
            return Ok(vocabulary);
        }

        // Fallback: try manually constructing vocabulary from tokens and scores
        if let Some(GGUFValue::Array(tokens)) = model.get_metadata("tokenizer.ggml.tokens") {
            let scores = match model.get_metadata("tokenizer.ggml.scores") {
                Some(GGUFValue::Array(scores)) => scores,
                _ => {
                    // If no scores are available, use default scores (0.0)
                    &Vec::<GGUFValue>::new()
                }
            };

            let mut vocabulary = Vec::with_capacity(tokens.len());
            for (i, token_value) in tokens.iter().enumerate() {
                if let GGUFValue::String(token_str) = token_value {
                    let score = if i < scores.len() {
                        if let GGUFValue::F32(score_val) = scores[i] {
                            score_val
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };

                    // Infer special tokens based on common patterns
                    let token_type = if token_str.starts_with("<") && token_str.ends_with(">") {
                        1 // Special token
                    } else {
                        0 // Normal token
                    };

                    vocabulary.push(TokenInfo {
                        token: token_str.clone(),
                        score,
                        token_type,
                    });
                }
            }

            if !vocabulary.is_empty() {
                return Ok(vocabulary);
            }
        }

        // Try different keys for Qwen models
        if let Some(GGUFValue::Array(tokens)) = model.get_metadata("tokenizer.qwen2.tokens") {
            let mut vocabulary = Vec::with_capacity(tokens.len());
            for token_value in tokens.iter() {
                if let GGUFValue::String(token_str) = token_value {
                    // Infer special tokens based on common patterns
                    let token_type = if token_str.starts_with("<") && token_str.ends_with(">") {
                        1 // Special token
                    } else {
                        0 // Normal token
                    };

                    vocabulary.push(TokenInfo {
                        token: token_str.clone(),
                        score: 0.0,
                        token_type,
                    });
                }
            }

            if !vocabulary.is_empty() {
                return Ok(vocabulary);
            }
        }

        // For Qwen models specifically, try to create a fallback vocabulary
        if model.get_architecture() == GGUFArchitecture::Qwen2 {
            // Create basic vocabulary for Qwen
            println!("Using fallback vocabulary for Qwen2 model");

            // Create a basic vocabulary with common tokens
            let mut vocab = Vec::new();
            // Add special tokens
            vocab.push(TokenInfo {
                token: "<pad>".to_string(),
                score: 0.0,
                token_type: 1, // Special token
            });
            vocab.push(TokenInfo {
                token: "<bos>".to_string(),
                score: 0.0,
                token_type: 1, // Special token
            });
            vocab.push(TokenInfo {
                token: "<eos>".to_string(),
                score: 0.0,
                token_type: 1, // Special token
            });

            // Add basic ASCII characters
            for i in 32..127 {
                vocab.push(TokenInfo {
                    token: std::char::from_u32(i).unwrap().to_string(),
                    score: 0.0,
                    token_type: 0, // Normal token
                });
            }

            return Ok(vocab);
        }

        // Generic fallback for any model - create a minimal vocabulary
        if model.tensors.len() > 0 {
            let mut vocab = Vec::new();
            // Add special tokens that most models have
            vocab.push(TokenInfo {
                token: "<unk>".to_string(),
                score: 0.0,
                token_type: 1, // Special token
            });
            vocab.push(TokenInfo {
                token: "<s>".to_string(),
                score: 0.0,
                token_type: 1, // Special token
            });
            vocab.push(TokenInfo {
                token: "</s>".to_string(),
                score: 0.0,
                token_type: 1, // Special token
            });

            // Add basic ASCII characters
            for i in 32..127 {
                vocab.push(TokenInfo {
                    token: std::char::from_u32(i).unwrap().to_string(),
                    score: 0.0,
                    token_type: 0, // Normal token
                });
            }

            return Ok(vocab);
        }

        Err(format!(
            "Failed to extract vocabulary from GGUF model (architecture: {}). The model may not contain tokenizer data or using an unsupported format.",
            model.get_architecture().as_str()
        ))
    }

    /// Extract special token IDs from model metadata
    fn extract_special_tokens(
        model: &GGUFModel,
        architecture: &GGUFArchitecture,
    ) -> (u32, u32, Option<u32>) {
        // Default values based on architecture
        let (default_bos, default_eos, default_unk) = match architecture {
            GGUFArchitecture::Llama => (1, 2, Some(0)),
            GGUFArchitecture::Qwen2 => (1, 2, Some(0)),
            GGUFArchitecture::Phi2 | GGUFArchitecture::Phi3 => (1, 2, Some(0)),
            _ => (1, 2, Some(0)), // Generic defaults
        };

        // Try to extract BOS token ID
        let bos_id = match model.get_metadata("tokenizer.ggml.bos_token_id") {
            Some(GGUFValue::U32(id)) => *id,
            Some(GGUFValue::I32(id)) => *id as u32,
            _ => match model.get_metadata("tokenizer.bos_token_id") {
                Some(GGUFValue::U32(id)) => *id,
                Some(GGUFValue::I32(id)) => *id as u32,
                _ => default_bos,
            },
        };

        // Try to extract EOS token ID
        let eos_id = match model.get_metadata("tokenizer.ggml.eos_token_id") {
            Some(GGUFValue::U32(id)) => *id,
            Some(GGUFValue::I32(id)) => *id as u32,
            _ => match model.get_metadata("tokenizer.eos_token_id") {
                Some(GGUFValue::U32(id)) => *id,
                Some(GGUFValue::I32(id)) => *id as u32,
                _ => default_eos,
            },
        };

        // Try to extract UNK token ID
        let unk_id = match model.get_metadata("tokenizer.ggml.unknown_token_id") {
            Some(GGUFValue::U32(id)) => Some(*id),
            Some(GGUFValue::I32(id)) => Some(*id as u32),
            _ => match model.get_metadata("tokenizer.unknown_token_id") {
                Some(GGUFValue::U32(id)) => Some(*id),
                Some(GGUFValue::I32(id)) => Some(*id as u32),
                _ => default_unk,
            },
        };

        (bos_id, eos_id, unk_id)
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Check if a token is a special token
    pub fn is_special_token(&self, token_id: u32) -> bool {
        if token_id as usize >= self.vocabulary.len() {
            return false;
        }
        self.vocabulary[token_id as usize].token_type != 0
    }

    /// Split text into smaller units for tokenization
    /// The implementation depends on the model architecture
    fn split_text(&self, text: &str) -> Vec<String> {
        match self.architecture {
            GGUFArchitecture::Llama | GGUFArchitecture::Phi2 | GGUFArchitecture::Phi3 => {
                // For Llama and Phi models, replace spaces with ▁ (U+2581)
                // This follows the SentencePiece (Unigram) tokenization approach
                let mut result = Vec::new();
                let text = text.replace(' ', "▁");
                let text = format!("▁{}", text.trim_start());

                // For simplicity, we'll use a basic character-based split,
                // which isn't as sophisticated as SentencePiece but works for basic usage
                let mut current = String::new();
                for c in text.chars() {
                    current.push(c);
                    if self.token_to_id.contains_key(&current) {
                        result.push(current.clone());
                        current.clear();
                    }
                }

                if !current.is_empty() {
                    // Handle any remaining characters
                    for c in current.chars() {
                        result.push(c.to_string());
                    }
                }

                result
            }
            GGUFArchitecture::Qwen2 => {
                // Qwen2 typically uses a byte-level BPE tokenizer similar to GPT-2
                let mut result = Vec::new();

                // First try word-level splitting
                for word in text.split_whitespace() {
                    if self.token_to_id.contains_key(word) {
                        // If the word is in the vocabulary, use it directly
                        result.push(word.to_string());
                    } else {
                        // Otherwise, fall back to character-level tokenization
                        for c in word.chars() {
                            result.push(c.to_string());
                        }
                        // Add space after each word except the last one
                        if !word.is_empty() {
                            result.push(" ".to_string());
                        }
                    }
                }

                result
            }
            GGUFArchitecture::Gpt2 => {
                // For GPT-2 models, implement a simple character-level split
                // This is a simplified approach - proper BPE would be more complex
                text.chars().map(|c| c.to_string()).collect()
            }
            _ => {
                // For other architectures, fall back to character-level tokenization
                text.chars().map(|c| c.to_string()).collect()
            }
        }
    }

    /// Merge tokens based on the vocabulary
    fn merge_tokens(&self, tokens: Vec<String>) -> Vec<u32> {
        let mut result = Vec::new();

        // This is a simplified greedy approach
        let mut i = 0;
        while i < tokens.len() {
            // Try to match the longest token sequence
            let mut best_length = 0;
            let mut best_id = self.unk_id.unwrap_or(0);

            for j in 1..=std::cmp::min(10, tokens.len() - i) {
                // Limit lookahead to 10 tokens for performance
                let sub_token = tokens[i..i + j].join("");
                if let Some(&id) = self.token_to_id.get(&sub_token) {
                    best_length = j;
                    best_id = id;
                }
            }

            if best_length > 0 {
                result.push(best_id);
                i += best_length;
            } else {
                // If no match found, use UNK token and advance
                result.push(self.unk_id.unwrap_or(0));
                i += 1;
            }
        }

        result
    }
}

impl Tokenizer for GGUFTokenizer {
    fn new(tokenizer_path: &str) -> Result<Self, String> {
        // Load the GGUF model
        println!("Loading GGUF model from: {}", tokenizer_path);
        let model = GGUFModel::load(tokenizer_path)
            .map_err(|e| format!("Failed to load GGUF model: {}", e))?;

        println!(
            "Detected model architecture: {}",
            model.get_architecture().as_str()
        );

        // Create the tokenizer from the model
        Self::from_gguf_model(&model)
    }

    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        // Split the text based on architecture-specific rules
        let tokens = self.split_text(text);

        // Merge tokens based on vocabulary
        let mut result = self.merge_tokens(tokens);

        // Add BOS/EOS tokens if requested
        if bos {
            result.insert(0, self.bos_id);
        }

        if eos {
            result.push(self.eos_id);
        }

        result
    }

    fn decode(&self, tokens: Vec<u32>) -> Result<String, String> {
        let mut result = String::new();
        let mut prev_was_special = false;

        for &token_id in &tokens {
            // Find the token with this ID
            if token_id as usize >= self.vocabulary.len() {
                return Err(format!(
                    "Token ID {} is out of range (vocab size: {})",
                    token_id,
                    self.vocabulary.len()
                ));
            }

            let token_info = &self.vocabulary[token_id as usize];
            let is_special = token_info.token_type != 0;

            // Handle tokens based on model architecture and token type
            if self.architecture == GGUFArchitecture::Llama
                || self.architecture == GGUFArchitecture::Phi2
                || self.architecture == GGUFArchitecture::Phi3
            {
                if is_special {
                    // Skip adding special tokens to the output if they're system tokens
                    // but keep user-facing special tokens like [USER], [ASSISTANT]
                    if token_info.token == "<s>"
                        || token_info.token == "</s>"
                        || token_info.token == "<pad>"
                        || token_info.token == "<unk>"
                    {
                        prev_was_special = true;
                        continue;
                    }
                    // Otherwise, add it normally
                    result.push_str(&token_info.token);
                } else if token_info.token.starts_with('▁') {
                    // If previous token was special, don't add an extra space
                    if prev_was_special {
                        result.push_str(&token_info.token[1..]);
                    } else {
                        // Replace the leading special char with space
                        result.push(' ');
                        result.push_str(&token_info.token[1..]);
                    }
                } else {
                    result.push_str(&token_info.token);
                }
            } else if is_special {
                // For other models, handle special tokens
                // Skip most system special tokens but keep user-facing ones
                if token_info.token == "<s>"
                    || token_info.token == "</s>"
                    || token_info.token == "<pad>"
                    || token_info.token == "<unk>"
                {
                    prev_was_special = true;
                    continue;
                }
                result.push_str(&token_info.token);
            } else {
                // Normal token for other models
                result.push_str(&token_info.token);
            }

            prev_was_special = is_special;
        }

        // Clean up the result
        // For Llama models, convert leading space to no space and ▁ to spaces
        if self.architecture == GGUFArchitecture::Llama
            || self.architecture == GGUFArchitecture::Phi2
            || self.architecture == GGUFArchitecture::Phi3
        {
            let result = result.trim_start().to_string();
            let result = result.replace('▁', " ");
            Ok(result)
        } else {
            Ok(result)
        }
    }

    fn bos_id(&self) -> u32 {
        self.bos_id
    }

    fn eos_id(&self) -> u32 {
        self.eos_id
    }

    fn stop_ids(&self) -> Vec<u32> {
        vec![self.eos_id]
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     // Test loading a tokenizer
//     #[test]
//     fn test_load_tokenizer() {
//         // This test requires a real GGUF model file
//         // Uncomment and point to a real model to test
//         let model_path = "/Volumes/Ollama/qwen2.5-0.5b-instruct-fp16.gguf";
//         let tokenizer = GGUFTokenizer::new(model_path);
//         assert!(tokenizer.is_ok());

//         if let Ok(tokenizer) = tokenizer {
//             // Test basic encoding and decoding
//             let text = "Hello, world!";
//             let tokens = tokenizer.encode(text, false, false);
//             let decoded = tokenizer.decode(tokens).unwrap();
//             println!("Original: '{}', Decoded: '{}'", text, decoded);

//             // Test with special tokens
//             let tokens_with_special = tokenizer.encode(text, true, true);
//             let decoded_with_special = tokenizer.decode(tokens_with_special).unwrap();
//             println!(
//                 "With special tokens - Original: '{}', Decoded: '{}'",
//                 text, decoded_with_special
//             );
//         }
//     }
// }
